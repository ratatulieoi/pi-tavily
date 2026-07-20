/**
 * pi-tavily — Native Tavily web search/extract/crawl/map/research extension for pi
 *
 * Registers 6 tools that the LLM can call directly:
 *   tavily_search   — Web search with depth/topic/domain filtering
 *   tavily_extract  — Clean content extraction from URLs
 *   tavily_crawl    — Website crawling with semantic filtering
 *   tavily_map      — Site structure discovery (URL mapping)
 *   tavily_research — Deep AI-powered research with cited reports
 *   tavily_usage    — Usage and credit totals for every configured key
 *
 * Commands:
 *   /tavily-key     — Validate and add an API key
 *   /tavily-usage   — Show usage for every configured key
 *
 * Features:
 *   - Auto-discovers TAVILY_API_KEY from env or ~/.pi/agent/tavily.json
 *   - Key rotation across multiple keys (round-robin) for max throughput
 *   - Abort-aware (respects ctx.signal / Esc to cancel)
 *   - Streaming progress updates via onUpdate
 */

import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateHead,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import * as os from "node:os";
import { setTimeout as delay } from "node:timers/promises";

// ─── Cache directory for full content ────────────────────────────

const CACHE_DIR = path.join(os.tmpdir(), `pi-tavily-cache-${process.getuid?.() ?? "user"}`);
fs.mkdirSync(CACHE_DIR, { recursive: true, mode: 0o700 });
fs.chmodSync(CACHE_DIR, 0o700);

function cacheContent(prefix: string, content: string): string {
  const hash = crypto.createHash("sha256").update(content).digest("hex").slice(0, 12);
  const filename = `${prefix}-${hash}.md`;
  const filepath = path.join(CACHE_DIR, filename);
  fs.writeFileSync(filepath, content, { mode: 0o600 });
  fs.chmodSync(filepath, 0o600);
  return filepath;
}

function limitOutput(
  prefix: string,
  content: string,
  cachePath?: string
): { text: string; cachePath?: string } {
  const truncated = truncateHead(content, {
    maxBytes: DEFAULT_MAX_BYTES,
    maxLines: DEFAULT_MAX_LINES,
  });
  if (!truncated.truncated) return { text: content, cachePath };

  const fullPath = cachePath ?? cacheContent(prefix, content);
  const notice = `[Output truncated: ${truncated.outputLines} of ${truncated.totalLines} lines (${formatSize(truncated.outputBytes)} of ${formatSize(truncated.totalBytes)}). Full output saved to: ${fullPath}]`;
  return { text: `${truncated.content}\n\n${notice}`, cachePath: fullPath };
}

function firstLine(text: string, maxChars = 180): string {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length <= maxChars) return cleaned;
  return cleaned.slice(0, maxChars).trimEnd() + "…";
}

function shortUrl(url: string): string {
  try {
    const u = new URL(url);
    const path = u.pathname.length > 30 ? u.pathname.slice(0, 27) + "…" : u.pathname;
    return u.hostname + (path === "/" ? "" : path);
  } catch {
    return url;
  }
}

// ─── Key management ───────────────────────────────────────────────

const TAVILY_BASE_URL = "https://api.tavily.com";

interface KeyPool {
  keys: string[];
  cycle(): string[];
}

function createKeyPool(keys: string[]): KeyPool {
  let idx = 0;
  return {
    keys,
    cycle() {
      if (keys.length === 0) throw new Error("No Tavily API keys configured");
      const start = idx++ % keys.length;
      return keys.map((_, offset) => keys[(start + offset) % keys.length]);
    },
  };
}

function getConfigPath(): string {
  return path.join(
    process.env.PI_CODING_AGENT_DIR || path.join(process.env.HOME || "/home/glam", ".pi/agent"),
    "tavily.json"
  );
}

function keysFromConfig(config: Record<string, unknown>): string[] {
  const keys = [
    ...(typeof config.apiKey === "string" ? [config.apiKey] : []),
    ...(Array.isArray(config.apiKeys) ? config.apiKeys : []),
    ...(Array.isArray(config.keys) ? config.keys : []),
  ];
  return [...new Set(keys.filter((key): key is string => typeof key === "string" && key.trim().length > 0).map((key) => key.trim()))];
}

interface ConfigState {
  keys: string[];
  error?: string;
}

function getConfigState(): ConfigState {
  const configPath = getConfigPath();
  if (!fs.existsSync(configPath)) return { keys: [] };
  try {
    return { keys: keysFromConfig(JSON.parse(fs.readFileSync(configPath, "utf-8"))) };
  } catch {
    return { keys: [], error: "Could not read ~/.pi/agent/tavily.json: invalid JSON." };
  }
}

function discoverKeys(): string[] {
  const envKeys = (process.env.TAVILY_API_KEY ?? "")
    .split(",")
    .map((key) => key.trim())
    .filter(Boolean);
  return [...new Set([...envKeys, ...getConfigState().keys])];
}

function saveKey(key: string): number {
  const configPath = getConfigPath();
  let config: Record<string, unknown> = {};
  if (fs.existsSync(configPath)) {
    try {
      config = JSON.parse(fs.readFileSync(configPath, "utf-8")) as Record<string, unknown>;
    } catch {
      throw new Error("Could not read ~/.pi/agent/tavily.json: invalid JSON.");
    }
    fs.chmodSync(configPath, 0o600);
  }
  const keys = keysFromConfig(config);
  if (keys.includes(key)) return keys.length;

  const nextConfig = { ...config, keys: [...keys, key] };
  delete nextConfig.apiKey;
  delete nextConfig.apiKeys;

  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  const tempPath = `${configPath}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(tempPath, `${JSON.stringify(nextConfig, null, 2)}\n`, { mode: 0o600 });
    fs.renameSync(tempPath, configPath);
    fs.chmodSync(configPath, 0o600);
  } finally {
    if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
  }

  return keys.length + 1;
}

// ─── API helpers ──────────────────────────────────────────────────

class TavilyHttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterMs?: number
  ) {
    super(message);
  }
}

function parseRetryAfter(value: string | null): number | undefined {
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}

async function tavilyRequest(
  endpoint: string,
  body: Record<string, unknown>,
  pool: KeyPool,
  signal?: AbortSignal,
  options: { method?: "POST" | "GET"; key?: string; timeoutMs?: number } = {}
): Promise<{ data: unknown; key: string }> {
  const method = options.method ?? "POST";
  const timeoutMs = options.timeoutMs ?? 30_000;
  const deadline = Date.now() + timeoutMs;
  const candidateKeys = options.key ? [options.key] : pool.cycle();
  let lastError: TavilyHttpError | undefined;

  for (const key of candidateKeys) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw new Error(`Tavily ${endpoint} timed out after ${timeoutMs / 1000} seconds`);
    }
    const timeoutSignal = AbortSignal.timeout(remainingMs);
    const requestSignal = signal
      ? AbortSignal.any([signal, timeoutSignal])
      : timeoutSignal;
    const fetchOptions: RequestInit = {
      method,
      headers: { Authorization: `Bearer ${key}` },
      signal: requestSignal,
    };
    if (method === "POST") {
      fetchOptions.headers = { ...fetchOptions.headers, "Content-Type": "application/json" };
      fetchOptions.body = JSON.stringify(body);
    }

    let response: Response;
    try {
      response = await fetch(`${TAVILY_BASE_URL}${endpoint}`, fetchOptions);
      if (response.ok) return { data: await response.json(), key };
    } catch (error) {
      if (signal?.aborted) throw new Error(`Tavily ${endpoint} cancelled`, { cause: error });
      if (timeoutSignal.aborted) {
        throw new Error(`Tavily ${endpoint} timed out after ${timeoutMs / 1000} seconds`, { cause: error });
      }
      throw new Error(`Tavily ${endpoint} network failure`, { cause: error });
    }

    let message = `Tavily ${endpoint} failed: HTTP ${response.status}`;
    try {
      const errorBody = (await response.json()) as {
        detail?: string | { error?: string };
        message?: string;
      };
      const detail = typeof errorBody.detail === "string"
        ? errorBody.detail
        : errorBody.detail?.error;
      if (detail) message += ` — ${detail}`;
      else if (errorBody.message) message += ` — ${errorBody.message}`;
    } catch {
      // Response body was not JSON.
    }

    const retryAfterMs = parseRetryAfter(response.headers.get("retry-after"));
    if (response.status === 429 && retryAfterMs !== undefined) {
      message += ` — retry after ${Math.ceil(retryAfterMs / 1000)}s`;
    }
    const error = new TavilyHttpError(message, response.status, retryAfterMs);
    const canRotate = !options.key && [401, 429, 432, 433].includes(response.status);
    if (!canRotate) throw error;
    lastError = error;
  }

  throw lastError ?? new Error(`Tavily ${endpoint} failed: no API keys available`);
}

async function wait(ms: number, signal?: AbortSignal): Promise<void> {
  try {
    await delay(ms, undefined, signal ? { signal } : undefined);
  } catch (error) {
    if (signal?.aborted) throw new Error("Research cancelled", { cause: error });
    throw error;
  }
}

interface TavilyUsage {
  usage?: number;
  limit?: number | null;
  search_usage?: number;
  extract_usage?: number;
  crawl_usage?: number;
  map_usage?: number;
  research_usage?: number;
}

interface UsageResponse {
  key?: TavilyUsage;
  account?: TavilyUsage & {
    current_plan?: string;
    plan_usage?: number;
    plan_limit?: number | null;
    paygo_usage?: number;
    paygo_limit?: number | null;
  };
}

interface KeyUsageResult {
  index: number;
  usage?: UsageResponse;
  error?: string;
}

async function getUsageForKey(key: string, signal?: AbortSignal): Promise<UsageResponse> {
  const pool = createKeyPool([key]);
  const { data } = await tavilyRequest("/usage", {}, pool, signal, {
    method: "GET",
    key,
    timeoutMs: 15_000,
  });
  return data as UsageResponse;
}

async function getAllUsage(keys: string[], signal?: AbortSignal): Promise<KeyUsageResult[]> {
  if (keys.length === 0) throw new Error("No Tavily API keys configured");
  return Promise.all(keys.map(async (key, index) => {
    try {
      return { index: index + 1, usage: await getUsageForKey(key, signal) };
    } catch (error) {
      if (signal?.aborted) throw error;
      return { index: index + 1, error: error instanceof Error ? error.message : String(error) };
    }
  }));
}

function formatUsage(results: KeyUsageResult[]): string {
  let totalLimit = 0;
  let remaining = 0;

  for (const result of results) {
    if (!result.usage) throw new Error(`Could not check Tavily key ${result.index}: ${result.error}`);
    const key = result.usage.key ?? {};
    const account = result.usage.account ?? {};
    const used = key.usage ?? account.plan_usage ?? 0;
    const limit = key.limit ?? account.plan_limit;
    if (limit == null) throw new Error(`Tavily key ${result.index} has no usage limit`);
    totalLimit += limit;
    remaining += Math.max(0, limit - used);
  }

  return [
    `Keys: ${results.length}`,
    `Total limit: ${totalLimit.toLocaleString()} credits`,
    `Remaining: ${remaining.toLocaleString()} credits`,
  ].join("\n");
}

// ─── Response formatting ──────────────────────────────────────────

interface SearchResult {
  title: string;
  url: string;
  content: string;
  score: number;
  raw_content?: string | null;
}

function dedupeSearchResults(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    let normalized = result.url;
    try {
      const url = new URL(result.url);
      url.hash = "";
      for (const key of [...url.searchParams.keys()]) {
        if (key.startsWith("utm_") || ["ref", "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid"].includes(key)) {
          url.searchParams.delete(key);
        }
      }
      url.searchParams.sort();
      url.pathname = url.pathname.replace(/\/+$/, "") || "/";
      normalized = url.toString();
    } catch {
      // Keep malformed URLs distinct by their original value.
    }
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

interface ExtractResult {
  url: string;
  raw_content: string | null;
  images?: Array<{ url: string; description?: string }>;
}

function formatSearchResponse(
  data: Record<string, unknown>,
  query: string
): { llmText: string; cachePath?: string } {
  const lines: string[] = [`**Query:** ${query}`, ""];
  const results = (data.results as SearchResult[]) || [];

  // 1. Compact result list — one line per source
  if (results.length > 0) {
    lines.push(`**Sources (${results.length}):**`);
    results.forEach((r, i) => {
      lines.push(`${i + 1}. [${shortUrl(r.url)}](${r.url}) ${firstLine(r.content)}`);
    });
  } else {
    lines.push("_No results found._");
  }

  // 2. Cache full raw content to disk if present, return file path to LLM
  const hasRaw = results.some((r) => r.raw_content);
  let cachePath: string | undefined;
  if (hasRaw) {
    const fullContent = results
      .map((r, i) => {
        const parts = [`## [${i + 1}] ${r.title}`, `URL: ${r.url}`, `Score: ${r.score}`, ""];
        if (r.raw_content) parts.push(r.raw_content);
        else parts.push(r.content);
        return parts.join("\n");
      })
      .join("\n\n---\n\n");
    cachePath = cacheContent("search", `# Tavily Search: ${query}\n\n${fullContent}`);
    lines.push("");
    lines.push(`_Full content cached: ${cachePath} (read this file if you need deep content)_`);
  }

  return { llmText: lines.join("\n"), cachePath };
}

function formatExtractResponse(
  data: Record<string, unknown>
): { llmText: string; cachePath?: string } {
  const results = (data.results as ExtractResult[]) || [];
  const failed = (data.failed_results as Array<{ url: string; error?: string }>) || [];
  const lines: string[] = [];

  results.forEach((result, index) => {
    const content = result.raw_content || "(empty)";
    const preview = content.length <= 400 ? content : `${content.slice(0, 400)}…`;
    lines.push(`**[${index + 1}] ${shortUrl(result.url)}**`);
    lines.push(preview);
    lines.push("");
  });

  if (results.length === 0) lines.push("_No content extracted._");
  if (failed.length > 0) {
    lines.push(`_Failed: ${failed.map((result) => result.error ? `${result.url} (${result.error})` : result.url).join(", ")}_`);
  }

  let cachePath: string | undefined;
  if (results.some((result) => (result.raw_content?.length || 0) > 400)) {
    const fullContent = results
      .map((result, index) => `## [${index + 1}] ${result.url}\n\n${result.raw_content || "(empty)"}`)
      .join("\n\n---\n\n");
    cachePath = cacheContent("extract", fullContent);
    lines.push("");
    lines.push(`_Full content cached: ${cachePath} (read this file for complete extraction)_`);
  }

  return { llmText: lines.join("\n"), cachePath };
}

function formatCrawlResponse(
  data: Record<string, unknown>
): { llmText: string; cachePath?: string } {
  const results = (data.results as Array<{ url: string; raw_content?: string | null }>) || [];
  const lines: string[] = [];

  lines.push(`**Crawled ${results.length} pages:**`);
  results.forEach((r, i) => {
    const preview = r.raw_content ? firstLine(r.raw_content, 150) : "(no content)";
    lines.push(`${i + 1}. [${shortUrl(r.url)}] ${preview}`);
  });

  let cachePath: string | undefined;
  if (results.some((r) => r.raw_content)) {
    const fullContent = results
      .map((r, i) => `## [${i + 1}] ${r.url}\n\n${r.raw_content || "(empty)"}`)
      .join("\n\n---\n\n");
    cachePath = cacheContent("crawl", fullContent);
    lines.push("");
    lines.push(`_Full crawl cached: ${cachePath}_`);
  }

  return { llmText: lines.join("\n"), cachePath };
}

function formatMapResponse(
  data: Record<string, unknown>
): { llmText: string; cachePath?: string } {
  const urls = (data.results as string[]) || [];
  const lines: string[] = [];

  lines.push(`**Found ${urls.length} URLs.**`);

  // If small list, show all; if large, show first 20 + cache full list
  let cachePath: string | undefined;
  if (urls.length <= 20) {
    urls.forEach((u) => lines.push(`- ${u}`));
  } else {
    urls.slice(0, 20).forEach((u) => lines.push(`- ${u}`));
    lines.push(`... and ${urls.length - 20} more.`);
    cachePath = cacheContent("map", urls.join("\n"));
    lines.push("");
    lines.push(`_Full URL list cached: ${cachePath}_`);
  }

  return { llmText: lines.join("\n"), cachePath };
}

// ─── Extension ────────────────────────────────────────────────────

export default async function(pi: ExtensionAPI) {
  const configState = getConfigState();
  const keys = discoverKeys();
  const pool = createKeyPool(keys);

  // ─── Commands ─────────────────────────────────────────────────

  pi.registerCommand("tavily-key", {
    description: "Validate and add a Tavily API key",
    handler: async (args, ctx) => {
      if (!args.trim() && !ctx.hasUI) {
        ctx.ui.notify("Usage: /tavily-key <key>", "error");
        return;
      }

      const key = (args.trim() || await ctx.ui.input("Add Tavily API key", "tvly-...") || "").trim();
      if (!key) {
        ctx.ui.notify("Cancelled. No changes made.", "info");
        return;
      }
      if (discoverKeys().includes(key)) {
        ctx.ui.notify("Key already exists. No changes made.", "warning");
        return;
      }

      try {
        await getUsageForKey(key);
      } catch (error) {
        const message = error instanceof TavilyHttpError && error.status === 401
          ? "Invalid Tavily key. No changes made."
          : "Could not validate Tavily key. No changes made.";
        ctx.ui.notify(message, "error");
        return;
      }

      try {
        saveKey(key);
      } catch (error) {
        const message = error instanceof Error && error.message.includes("invalid JSON")
          ? `${error.message}\nKey was not added.`
          : "Could not save Tavily key. Key was not added.";
        ctx.ui.notify(message, "error");
        return;
      }

      ctx.ui.notify("Tavily key added. Reloading...", "info");
      try {
        await ctx.reload();
      } catch {
        ctx.ui.notify("Tavily key was added, but reload failed. Run /reload.", "error");
      }
    },
  });

  pi.registerCommand("tavily-usage", {
    description: "Show usage for every configured Tavily API key",
    handler: async (_args, ctx) => {
      ctx.ui.notify("Checking Tavily usage...", "info");
      try {
        ctx.ui.notify(formatUsage(await getAllUsage(discoverKeys())), "info");
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  // ─── tavily_usage ─────────────────────────────────────────────

  pi.registerTool({
    name: "tavily_usage",
    label: "Tavily Usage",
    description: "Get credit usage and limits for every configured Tavily API key, with per-endpoint and combined totals.",
    promptSnippet: "Check Tavily API credit usage across configured keys",
    promptGuidelines: [
      "Use tavily_usage when the user asks about Tavily credits, limits, remaining usage, or configured key health.",
    ],
    parameters: Type.Object({}),

    async execute(_toolCallId, _params, signal, onUpdate) {
      const currentKeys = discoverKeys();
      onUpdate?.({ content: [{ type: "text", text: `Checking usage for ${currentKeys.length} Tavily key(s)...` }] });
      const results = await getAllUsage(currentKeys, signal);
      return {
        content: [{ type: "text", text: formatUsage(results) }],
        details: {
          keyCount: currentKeys.length,
          successCount: results.filter((result) => result.usage).length,
          failedCount: results.filter((result) => result.error).length,
        },
      };
    },
  });

  // ─── tavily_search ────────────────────────────────────────────

  pi.registerTool({
    name: "tavily_search",
    label: "Tavily Search",
    description:
      "Search the web using Tavily. Returns ranked results with content snippets and relevance scores. Supports search depth (basic/advanced/fast/ultra-fast), topic filtering (general/news/finance), and time ranges.",
    promptSnippet: "Search the web for real-time information (broad fetch, compact output)",
    promptGuidelines: [
      "Use tavily_search for current web info, news, or facts not in your training data. It returns 5 results by default in a compact summary. Use 'advanced' depth + include_raw_content for deep dives.",
      "tavily_search returns one-line summaries per source plus URLs. Full raw content is cached to a file path — use the read tool on that file only if you need deep content.",
      "Use tavily_search with topic='news' for recent events, topic='finance' for financial data.",
      "After tavily_search, if you need full content from specific URLs, pass those URLs to tavily_extract — don't re-search.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      search_depth: StringEnum(["basic", "advanced", "fast", "ultra-fast"] as const, {
        description:
          "Search depth: 'basic' (balanced, 1 credit), 'advanced' (highest relevance, 2 credits), 'fast' (lower latency, 1 credit), 'ultra-fast' (minimal latency, 1 credit). Default: basic",
        default: "basic",
      }),
      topic: StringEnum(["general", "news", "finance"] as const, {
        description: "Search topic: 'general' for broad searches, 'news' for real-time updates, 'finance' for financial data. Default: general",
        default: "general",
      }),
      max_results: Type.Optional(Type.Number({ description: "Max results to return (1-20). Default: 5", minimum: 1, maximum: 20, default: 5 })),
      time_range: Type.Optional(
        StringEnum(["day", "week", "month", "year"] as const, {
          description: "Filter results by time range from current date. Useful for recent content.",
        })
      ),
      include_raw_content: Type.Optional(
        StringEnum(["false", "markdown", "text"] as const, {
          description: "Include raw page content: 'markdown' for markdown format, 'text' for plain text. Default: false",
          default: "false",
        })
      ),
      chunks_per_source: Type.Optional(
        Type.Number({ description: "Max chunks per source (1-3). Only for 'advanced' depth. Default: 3", minimum: 1, maximum: 3 })
      ),
    }),

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: `🔍 Searching: "${params.query}"...` }] });

      // Official low-cost defaults: basic depth, 5 compact results, no raw content.
      const body: Record<string, unknown> = {
        query: params.query,
        search_depth: params.search_depth || "basic",
        topic: params.topic || "general",
        max_results: params.max_results ?? 5,
      };
      // Only include raw content if explicitly requested.
      if (params.include_raw_content && params.include_raw_content !== "false") {
        body.include_raw_content = params.include_raw_content === "true" ? "markdown" : params.include_raw_content;
      }

      // Remaining user overrides
      if (params.time_range !== undefined) body.time_range = params.time_range;
      if (params.chunks_per_source !== undefined) body.chunks_per_source = params.chunks_per_source;

      const { data: responseData } = await tavilyRequest("/search", body, pool, signal);
      const data = responseData as Record<string, unknown>;
      const results = dedupeSearchResults((data.results as SearchResult[]) || []);
      const formatted = formatSearchResponse({ ...data, results }, params.query);
      const output = limitOutput("search-output", formatted.llmText);

      return {
        content: [{ type: "text", text: output.text }],
        details: {
          query: params.query,
          resultCount: results.length,
          creditsUsed: (data.usage as Record<string, unknown>)?.credits,
          cachePath: output.cachePath ?? formatted.cachePath,
          urls: results.map((result) => result.url),
        },
      };
    },
  });

  // ─── tavily_extract ───────────────────────────────────────────

  pi.registerTool({
    name: "tavily_extract",
    label: "Tavily Extract",
    description:
      "Extract clean content from web page URLs. Returns markdown or text content, handles JavaScript-rendered pages. Supports query-focused extraction that reranks chunks by relevance.",
    promptSnippet: "Extract clean content from web URLs (preview + cached file)",
    promptGuidelines: [
      "Use tavily_extract to get actual content of specific URLs. Returns a short preview per URL + a cached file path with full extracted content. Read that file when you need the full text.",
      "Use tavily_extract with a query parameter to focus extraction on relevant chunks — reduces noise.",
      "Use tavily_extract after tavily_search to dive deep into URLs from the search result list.",
    ],
    parameters: Type.Object({
      urls: Type.Array(Type.String(), {
        description: "URLs to extract content from (1-20)",
        minItems: 1,
        maxItems: 20,
      }),
      query: Type.Optional(
        Type.String({ description: "Rerank extracted chunks by relevance to this query" })
      ),
      extract_depth: StringEnum(["basic", "advanced"] as const, {
        description: "'basic' (1 credit/5 successful URLs, fast) or 'advanced' (2 credits/5, more data and higher success). Default: basic",
        default: "basic",
      }),
      format: StringEnum(["markdown", "text"] as const, {
        description: "Output format: 'markdown' or 'text'. Default: markdown",
        default: "markdown",
      }),
      chunks_per_source: Type.Optional(
        Type.Number({ description: "Max relevant chunks per source (1-5, only with query). Default: 3", minimum: 1, maximum: 5 })
      ),
    }),

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: `📄 Extracting content from ${params.urls.length} URL(s)...` }] });

      const body: Record<string, unknown> = {
        urls: params.urls,
        extract_depth: params.extract_depth || "basic",
        format: params.format || "markdown",
      };

      if (params.query) body.query = params.query;
      if (params.chunks_per_source) body.chunks_per_source = params.chunks_per_source;

      const { data: responseData } = await tavilyRequest("/extract", body, pool, signal, {
        timeoutMs: params.extract_depth === "advanced" ? 35_000 : 15_000,
      });
      const data = responseData as Record<string, unknown>;
      const formatted = formatExtractResponse(data);
      const output = limitOutput("extract-output", formatted.llmText);

      return {
        content: [{ type: "text", text: output.text }],
        details: {
          urlCount: params.urls.length,
          successCount: ((data.results as unknown[]) || []).length,
          failedCount: ((data.failed_results as unknown[]) || []).length,
          cachePath: output.cachePath ?? formatted.cachePath,
        },
      };
    },
  });

  // ─── tavily_crawl ─────────────────────────────────────────────

  pi.registerTool({
    name: "tavily_crawl",
    label: "Tavily Crawl",
    description:
      "Crawl a website starting from a URL, discovering and extracting content from multiple pages. Supports semantic filtering with instructions, path/domain filtering, and depth/breadth control.",
    promptSnippet: "Crawl website pages and extract their content (cached to disk)",
    promptGuidelines: [
      "Use tavily_crawl when you need content from multiple pages of a site. Returns a one-line preview per page + cached file with full content.",
      "Use tavily_crawl with instructions to focus the crawl (e.g., 'Find all API docs').",
      "Prefer tavily_map + tavily_extract over tavily_crawl when you only need a few specific pages — it's cheaper and more targeted.",
    ],
    parameters: Type.Object({
      url: Type.String({ description: "The root URL to start crawling from" }),
      instructions: Type.Optional(
        Type.String({ description: "Natural language instructions to focus the crawl (e.g., 'Find all pages about authentication')" })
      ),
      max_depth: Type.Optional(
        Type.Number({ description: "Max crawl depth (1-5). Default: 1", minimum: 1, maximum: 5 })
      ),
      max_breadth: Type.Optional(
        Type.Number({ description: "Max links per page (1-500). Default: 20", minimum: 1, maximum: 500 })
      ),
      limit: Type.Optional(
        Type.Number({ description: "Total pages to process before stopping. Default: 50", minimum: 1 })
      ),
      select_paths: Type.Optional(
        Type.Array(Type.String(), { description: "Regex patterns to include only matching URL paths (e.g., '/docs/.*')" })
      ),
      exclude_paths: Type.Optional(
        Type.Array(Type.String(), { description: "Regex patterns to exclude matching URL paths (e.g., '/blog/.*')" })
      ),
      extract_depth: StringEnum(["basic", "advanced"] as const, {
        description: "Extraction depth per page. Default: basic",
        default: "basic",
      }),
      format: StringEnum(["markdown", "text"] as const, {
        description: "Content format. Default: markdown",
        default: "markdown",
      }),
      chunks_per_source: Type.Optional(
        Type.Number({ description: "Max relevant chunks per source (1-5, only with instructions). Default: 3", minimum: 1, maximum: 5 })
      ),
    }),

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: `🕷️ Crawling: ${params.url}...` }] });

      const body: Record<string, unknown> = {
        url: params.url,
        extract_depth: params.extract_depth || "basic",
        format: params.format || "markdown",
      };

      if (params.instructions) body.instructions = params.instructions;
      if (params.max_depth) body.max_depth = params.max_depth;
      if (params.max_breadth) body.max_breadth = params.max_breadth;
      if (params.limit) body.limit = params.limit;
      if (params.select_paths?.length) body.select_paths = params.select_paths;
      if (params.exclude_paths?.length) body.exclude_paths = params.exclude_paths;
      if (params.chunks_per_source) body.chunks_per_source = params.chunks_per_source;

      const { data: responseData } = await tavilyRequest("/crawl", body, pool, signal, {
        timeoutMs: 160_000,
      });
      const data = responseData as Record<string, unknown>;
      const formatted = formatCrawlResponse(data);
      const output = limitOutput("crawl-output", formatted.llmText);

      return {
        content: [{ type: "text", text: output.text }],
        details: {
          url: params.url,
          pageCount: ((data.results as unknown[]) || []).length,
          cachePath: output.cachePath ?? formatted.cachePath,
        },
      };
    },
  });

  // ─── tavily_map ───────────────────────────────────────────────

  pi.registerTool({
    name: "tavily_map",
    label: "Tavily Map",
    description:
      "Discover all URLs on a website without extracting content. Fast site structure mapping with natural language filtering instructions. Use before tavily_extract for efficient targeted extraction.",
    promptSnippet: "Discover URLs and site structure on a website (no content)",
    promptGuidelines: [
      "Use tavily_map to discover what pages exist on a site before extracting. Returns URL list only (no content).",
      "Standard workflow: tavily_map to find target URLs → tavily_extract on those URLs for full content. Cheaper than tavily_crawl.",
    ],
    parameters: Type.Object({
      url: Type.String({ description: "The root URL to map" }),
      instructions: Type.Optional(
        Type.String({ description: "Natural language instructions to filter discovered URLs (e.g., 'Find API docs')" })
      ),
      max_depth: Type.Optional(
        Type.Number({ description: "Max mapping depth (1-5). Default: 1", minimum: 1, maximum: 5 })
      ),
      limit: Type.Optional(
        Type.Number({ description: "Max URLs to discover. Default: 50", minimum: 1 })
      ),
      select_paths: Type.Optional(
        Type.Array(Type.String(), { description: "Regex patterns to include only matching URL paths" })
      ),
      exclude_paths: Type.Optional(
        Type.Array(Type.String(), { description: "Regex patterns to exclude matching URL paths" })
      ),
    }),

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: `🗺️ Mapping: ${params.url}...` }] });

      const body: Record<string, unknown> = {
        url: params.url,
      };

      if (params.instructions) body.instructions = params.instructions;
      if (params.max_depth) body.max_depth = params.max_depth;
      if (params.limit) body.limit = params.limit;
      if (params.select_paths?.length) body.select_paths = params.select_paths;
      if (params.exclude_paths?.length) body.exclude_paths = params.exclude_paths;

      const { data: responseData } = await tavilyRequest("/map", body, pool, signal, {
        timeoutMs: 160_000,
      });
      const data = responseData as Record<string, unknown>;
      const formatted = formatMapResponse(data);
      const output = limitOutput("map-output", formatted.llmText);

      return {
        content: [{ type: "text", text: output.text }],
        details: {
          url: params.url,
          urlCount: ((data.results as string[]) || []).length,
          cachePath: output.cachePath ?? formatted.cachePath,
        },
      };
    },
  });

  // ─── tavily_research ──────────────────────────────────────────

  pi.registerTool({
    name: "tavily_research",
    label: "Tavily Research",
    description:
      "Deep AI-powered research that gathers sources, analyzes them, and produces a cited report. Takes 30-120 seconds and supports mini/pro/auto models.",
    promptSnippet: "Conduct deep AI-powered research on a topic",
    promptGuidelines: [
      "Use tavily_research for comprehensive, multi-source research when you need a detailed cited report.",
      "Use tavily_research with model='mini' for targeted, efficient research on narrow topics.",
      "Use tavily_research with model='pro' for comprehensive research on complex, multi-domain topics.",
      "tavily_research takes 30-120 seconds — use it for thorough investigations, not quick lookups.",
      "For quick factual lookups, use tavily_search instead of tavily_research.",
    ],
    parameters: Type.Object({
      input: Type.String({ description: "The research question or topic to investigate" }),
      model: StringEnum(["mini", "pro", "auto"] as const, {
        description: "Research model: 'mini' (targeted), 'pro' (comprehensive), 'auto' (lets Tavily choose). Default: auto",
        default: "auto",
      }),
      citation_format: StringEnum(["numbered", "mla", "apa", "chicago"] as const, {
        description: "Citation format in report. Default: numbered",
        default: "numbered",
      }),
    }),

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: `🔬 Researching: "${params.input}"...\nThis may take 30-120 seconds.` }] });

      // Step 1: Initiate research
      const body: Record<string, unknown> = {
        input: params.input,
        model: params.model || "auto",
        citation_format: params.citation_format || "numbered",
      };

      // Tavily research tasks are key-scoped, so keep the exact key used to create it.
      const { data: initData, key: researchKey } = await tavilyRequest("/research", body, pool, signal);
      const initResponse = initData as Record<string, unknown>;
      const requestId = initResponse.request_id as string;

      if (!requestId) {
        // Direct response (no polling needed) or error
        const errDetail = (initResponse.detail as Record<string, unknown>)?.error;
        if (errDetail) throw new Error(`Research initiation failed: ${errDetail}`);
        const output = limitOutput("research-init", JSON.stringify(initResponse, null, 2));
        return {
          content: [{ type: "text", text: output.text }],
          details: { ...initResponse, cachePath: output.cachePath },
        };
      }

      onUpdate?.({ content: [{ type: "text", text: `🔬 Research started (ID: ${requestId}). Polling for results...` }] });

      // Step 2: Poll with the same key until completion or the wall-clock deadline.
      const pollInterval = 3000;
      const startedAt = Date.now();
      const deadline = startedAt + 180_000;

      while (Date.now() < deadline) {
        await wait(Math.min(pollInterval, deadline - Date.now()), signal);

        let statusData: Record<string, unknown>;
        try {
          const { data } = await tavilyRequest(`/research/${requestId}`, {}, pool, signal, {
            method: "GET",
            key: researchKey,
            timeoutMs: Math.min(15_000, Math.max(1, deadline - Date.now())),
          });
          statusData = data as Record<string, unknown>;
        } catch (error) {
          if (error instanceof TavilyHttpError && [429, 500, 502, 503, 504].includes(error.status)) {
            const retryDelay = Math.min(error.retryAfterMs ?? pollInterval, Math.max(0, deadline - Date.now()));
            onUpdate?.({ content: [{ type: "text", text: `🔬 Research poll delayed: ${error.message}` }] });
            if (retryDelay > 0) await wait(retryDelay, signal);
            continue;
          }
          throw error;
        }
        const status = statusData.status as string;

        if (status === "completed") {
          const content = typeof statusData.content === "string"
            ? statusData.content
            : JSON.stringify(statusData.content ?? "", null, 2);
          const sources = (statusData.sources as Array<{ url: string; title: string }>) || [];

          const lines: string[] = [];
          lines.push(`## Research Report: ${params.input}`);
          lines.push("");
          lines.push(content);
          if (sources.length > 0) {
            lines.push("");
            lines.push(`---`);
            lines.push(`**Sources (${sources.length}):**`);
            sources.forEach((s, i) => {
              lines.push(`${i + 1}. [${s.title || s.url}](${s.url})`);
            });
          }

          const output = limitOutput("research", lines.join("\n"));
          return {
            content: [{ type: "text", text: output.text }],
            details: {
              requestId,
              model: params.model,
              creditsUsed: (statusData.usage as Record<string, unknown>)?.credits,
              sourcesCount: sources.length,
              cachePath: output.cachePath,
            },
          };
        }

        if (status === "failed") {
          const errDetail = (statusData.detail as Record<string, unknown>)?.error || statusData.error;
          throw new Error(`Research failed: ${errDetail || "unknown error"}`);
        }

        // Still in progress — show elapsed time
        const elapsed = Math.round((Date.now() - startedAt) / 1000);
        onUpdate?.({ content: [{ type: "text", text: `🔬 Research in progress... (${elapsed}s elapsed)` }] });
      }

      throw new Error("Research timed out after 180 seconds");
    },
  });

  // ─── Startup notification ─────────────────────────────────────

  pi.on("session_start", (_event, ctx) => {
    if (configState.error) {
      ctx.ui.notify(configState.error, "warning");
    } else if (keys.length === 0) {
      ctx.ui.notify("pi-tavily: No API keys found. Run /tavily-key to add one.", "warning");
    } else {
      ctx.ui.notify(`pi-tavily ready: ${keys.length} keys`, "info");
    }
  });

}