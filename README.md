# pi-tavily

Native [Tavily](https://tavily.com) tools for [pi](https://github.com/earendil-works/pi-mono).

## Tools

- `tavily_search` — compact ranked web results with normalized URL deduplication
- `tavily_extract` — content extraction from known URLs
- `tavily_map` — website URL discovery
- `tavily_crawl` — website traversal and content extraction
- `tavily_research` — asynchronous cited research reports
- `tavily_usage` — total configured keys, credit limit, and remaining credits

Search returns only the query and short source snippets. Full raw content is stored in a private temporary cache when requested, keeping model context bounded.

## Install

```bash
pi install npm:pi-tavily
```

Or from GitHub:

```bash
pi install git:github.com/ratatulieoi/pi-tavily
```

## Configure

Run the interactive command:

```text
/tavily-key
```

Or create `~/.pi/agent/tavily.json`:

```json
{
  "keys": [
    "tvly-YOUR_API_KEY"
  ]
}
```

Multiple keys are rotated between requests. Research polling stays pinned to the key that created the task.

You can also set one or more comma-separated keys in `TAVILY_API_KEY`.

## Commands

- `/tavily-key` — validate and add a key, then reload pi
- `/tavily-usage` — show total keys, total credit limit, and remaining credits

## License

MIT
