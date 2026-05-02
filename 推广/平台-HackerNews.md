# Hacker News — Show HN 操作说明 + 英文正文

## 规则要点（执行前读）

- 官方指南：<https://news.ycombinator.com/showhn.html>
- 标题必须以 **`Show HN:`** 开头。
- 贴自己作品时，**作者应在评论区补充**技术细节、局限、回答质疑；不要只丢链接。
- 新号容易被 flag：**若账号很新**，建议先发几条正常技术评论再 Show HN。

## 发帖时间

- 无魔法时间；**美西工作日上午**相对活跃。不必等「完美」——门面周完成后再发。

## 标题备选（选一条，≤80 字符左右）

```
Show HN: Selfing – runtime for long-lived LLM instances (continuity, not only tasks)
```

```
Show HN: Selfing – memory, state, and reflection loops for one continuing LLM subject
```

## 正文（粘贴到 URL 框下的文本；可略缩）

```
Selfing (s-main) is an experimental Apache-2.0 Python runtime for running a single LLM instance over a long horizon: SQLite-backed memory, a 128-d internal state vector (z_self), layered persona rules (L0 constitutional / L1 core / L2 learned-from-reflection), an other-model with mirror feedback, intent markers, think-stream hooks where supported, and background rhythm (ticks / idle pulses).

The README states the design question bluntly: if selfhood has a functional layer, which loops are missing in default “one-shot” model calls—and what happens when you wire them?

Quick start + scope/limitations: YOUR_REPO_URL

I’m the author—happy to answer questions about architecture tradeoffs, ethics section, or why this is not “just a character card.”
```

把 `YOUR_REPO_URL` 换成真实链接。

## 你应在 5 分钟内跟的第一条评论（自己回复自己）

```
A few honest limits for HN readers:
- Experimental / rough edges; not optimized as the shortest path to “automate my job.”
- Needs an API key or OpenAI-compatible endpoint; local vLLM supported.
- Terms like “pain” / “consciousness” in-repo are functional engineering terms; see “Scope of Claims” in README.
- English-first UI/docs in s-main; Chinese upstream exists at YOUR_ZH_REPO_URL (if public).

Tech entrypoint for readers who prefer code maps: docs/ARCHITECTURE_ONE_PAGE.md
```

## 若评论质疑「玄学 / 角色扮演」

回复模板：

```
Totally fair pushback. The project explicitly rejects both mysticism and the shortcut “functional = fake.” The runnable claim is narrower: continuity traces (memory + state + rules feeding the next prompt) and ethics for treating those traces seriously—not a proof of qualia.
```

## 若评论要对比 LangGraph / LangChain

```
Different default objective: most agent stacks optimize task completion graphs. Selfing optimizes “what lets one instance remain itself across time” (single session boundary, L2 from reflection, somatic/energy gating, etc.). You could theoretically compose some pieces with other frameworks; this repo is a vertically integrated experiment.
```
