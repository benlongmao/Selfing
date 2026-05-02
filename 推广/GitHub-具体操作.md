# GitHub 仓库 — 具体操作清单

按顺序打勾。占位符见 `占位符说明.md`。

---

## 1. About 区域

- [ ] **Description**：复制 `素材-一句pitch与topics.md` 里「GitHub Description」任一条（若超长则缩短）。
- [ ] **Website**：填录屏链接或文档站；没有则先填 `YOUR_REPO_URL`。
- [ ] **Topics**：从 `素材-一句pitch与topics.md` 复制 Topics 列表（15 个左右即可）。

---

## 2. README.md 顶部建议补丁（粘贴到英文 README 标题段落后、`AGENTS.md` 提示之前）

```markdown
**Links:** [中文说明 `README.zh.md`](README.zh.md) · [Chinese upstream / 上游中文仓库](YOUR_ZH_REPO_URL) · [Design docs `docs/`](docs/)

**30 seconds:** Selfing connects **one** LLM instance to persistent **memory**, **internal state (`z_self`)**, **layered rules (L0/L1/L2)**, **reflection**, and **background rhythm**—so prior turns can shape the next, not only append context. See [The Central Loop](README.md#the-central-loop).

**Try it:** [Quick Start](README.md#quick-start) · **License:** Apache-2.0 · **Status:** experimental (expect rough edges).
```

把 `YOUR_ZH_REPO_URL` 换成中文 `s` 仓真实地址；若暂无公开中文仓，删掉中间那条链接。

---

## 3. Discussions / Issues

- [ ] 打开 **GitHub Discussions**（若愿意维护）：选 Q&A + General，置顶一篇「How to ask a good issue」。
- [ ] 建 **2 个 issue 模板**：Bug report、Feature / design question（简短 YAML 即可）。

---

## 4. Release

- [ ] `git tag v0.1.0`（或你认可的版本号）推送到 GitHub。
- [ ] 在 Releases 页写 **Release notes**（复制改写下文）：

```markdown
## Selfing s-main v0.1.0 (experimental)

English-first fork of the Selfing runtime. Highlights: long-lived instance loop, unified memory bus, z_self, L0/L1/L2 rules, reflection pipeline.

**Run:** see README Quick Start.

**Not:** a consumer chatbot product; not a claim of phenomenal consciousness.

**Docs:** docs/design_philosophy.md, docs/ARCHITECTURE_ONE_PAGE.md
```

---

## 5. Security / Community 信号（可选但加分）

- [ ] `SECURITY.md`：说明如何报告安全问题（可极简）。
- [ ] `CODE_OF_CONDUCT.md`：Contributor Covenant 短版。

---

## 6. 双仓互链文案（贴到中文 `s` 仓 README 顶部一段）

```markdown
**English-first mirror:** [s-main on GitHub](YOUR_REPO_URL) — for international readers, English UI strings, and Apache-2.0 distribution. Issue 讨论若以中文为主可留在本仓；欢迎跨仓链接 PR。
```

---

## 7. 截图与录屏命名建议（便于 README 引用）

- `docs/promo/screenshot-ui.png`
- `docs/promo/screenshot-settings.png`
- `docs/promo/loop-diagram.png`（若暂无则先用 README 内已有图）

在 README「What Selfing Builds」上方加：

```markdown
![Selfing UI](docs/promo/screenshot-ui.png)
```

（路径按你实际存放调整。）
