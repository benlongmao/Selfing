# Awesome 列表与外链 — 操作指南

## 策略

- 每周最多 **1～2 个** PR，避免被维护者视为 spam。
- PR 正文：**一句话价值 + 链接 + license + 维护状态**。

---

## 如何找目标列表

在 GitHub 搜索：

```
awesome llm stars:>500
awesome agents stars:>500
awesome local llm
awesome open source ai
```

点进仓库看 `contributing.md` 是否要求按字母序、是否只收「成熟」项目；**experimental** 列表更合适。

---

## PR 标题模板

```
Add Selfing (s-main): long-lived LLM runtime
```

---

## PR 正文模板（Markdown）

```markdown
[Selfing / s-main](YOUR_REPO_URL) is an **Apache-2.0** experimental Python runtime for **long-lived LLM instances**: persistent memory, internal state (`z_self`), layered rules (L0/L1/L2), reflection, and background rhythm—focused on **continuity** rather than only task completion.

- **Language / locale:** English-first docs & UI (`README.zh.md` for Chinese overview)
- **License:** Apache-2.0
- **Status:** experimental (see README expectations)

Disclaimer (from upstream README): does not claim phenomenal consciousness; uses functional definitions for terms like self / pain.
```

---

## 个人博客 / 外链

- 若你有旧文章流量：在文末加 **Update 2026-xx** 一节链到仓库。
- **Goodreads / 非技术站** 一般不投。

---

## Product Hunt（可选）

- 更偏产品；Selfing 偏研究运行时，**不是必选**。
- 若上：需要截图、一句 tagline、首评自己解释「for builders / researchers」。
