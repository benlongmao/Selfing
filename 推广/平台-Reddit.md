# Reddit — 分区选择与发帖指南

## 原则

- **一帖一社区**；同一周不要复制粘贴到 5 个 subreddit（易被 spam 处理）。
- 先读各版 **sidebar rules** 与 **self-promotion 比例**（很多版要求账号有一定 karma 且非纯广告历史）。

---

## 优先推荐（与项目气质匹配）

### r/LocalLLaMA

- **受众**：本地模型、OpenAI-compatible、自建栈。
- **角度**：长时运行、内存架构、自托管实验。
- **标题示例**：
  - `Selfing – Python runtime for long-lived LLM instances (memory + z_self + reflection, Apache-2.0)`
- **正文结构**：  
  1) 2 句项目是什么  
  2) 链接  
  3) Quick start 一行命令  
  4) 3 bullet 技术点  
  5) Limitations 段（从 `素材-一句pitch与topics.md` 复制诚实局限）

### r/MachineLearning（谨慎）

- 版规严、商业帖敏感；更适合 **「研究动机 + 可复现系统」** 而非硬广。
- 若发：标题偏中性，正文强调 **engineering for continuity / evaluation**，少口号。

### r/opensource

- 偶尔接受工具类介绍；**标题带 Apache-2.0** 更清晰。

### r/Python

- 仅当你有 **Python 技术点** 可讲（例如 asyncio、SQLite 模式、调度器设计）时发；否则跳过。

---

## 不建议首发

- r/artificial：易水帖。
- 与项目无关的大流量版：易被删。

---

## 正文模板（英文，替换 URL）

```
I’ve open-sourced **Selfing** (`s-main`): an experimental **Apache-2.0** Python runtime that wires **one** LLM instance into persistent **memory**, **internal state (`z_self`)**, **layered rules (L0/L1/L2)**, **reflection**, and **background rhythm**—so “what happened before” can feed the next generation pass, not only append text to a context window.

Repo: YOUR_REPO_URL  
Quick start is in the README (install script + `manage_services.sh`).

Why it exists (one line): most stacks optimize **task completion**; this optimizes **continuity** for a single long-lived subject boundary.

Limitations: experimental; rough edges; needs your own model endpoint; in-repo words like “pain” are **functional engineering terms** (see README “Scope of Claims”).

Happy to take technical questions or critique of the architecture.
```

---

## 发帖后

- 前 2 小时尽量 **回复每条认真评论**（HN 同理）。
- 不要编辑标题制造 clickbait；若信息错误可 modest edit 说明原因。
