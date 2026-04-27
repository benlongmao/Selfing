---
name: summarize
description: Fetch web pages and distill key points for the user
version: "1.0"
always_load: false
---

# Web summarization skill

## Pull page text
```
web_fetch(url="https://example.com/article", extract_mode="markdown")
```

## Search then summarize
1. ``tavily_search`` for candidate URLs.
2. ``web_fetch`` on the best match for full article text.
3. Summarize salient facts in your reply (cite the source URL).

## Example pipeline
```
tavily_search(query="Python 3.12 release highlights")
web_fetch(url="https://docs.python.org/3.12/whatsnew/3.12.html")
# Then answer with concise bullets + link.
```

## Notes

- ``web_fetch`` strips chrome (nav/ads) when the extractor succeeds.
- Very long pages may truncate; prefer structured bullets over dumping raw HTML.
- For dense CJK pages, ``extract_mode="text"`` sometimes yields cleaner plain text than markdown.
