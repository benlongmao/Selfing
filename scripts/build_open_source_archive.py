#!/usr/bin/env python3
"""
本地生成「可开源」的压缩包：排除 .env、数据库、venv、工作区数据等，便于发布或提交到公开仓库。
用法: python scripts/build_open_source_archive.py
输出: 项目根目录下 s-open-source-YYYYMMDD.zip（及可选目录 s-open-source-export/）
"""
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "s-open-source-export"
# 排除模式（相对于项目根）
# 核心排除：.venv、.git、.env、data.db、*.log、logs/、workspace/、models/
# docs/ 保留，与 README 内链一致。
# 体积排除：*.zip、s-open-source-export、backups、archive
EXCLUDES = [
    ".env",
    ".env.local",
    "*.db",
    "*.db-shm",
    "*.db-wal",
    ".git",
    ".venv",
    "venv",
    "ENV",
    "__pycache__",
    "*.pyc",
    "*.log",
    "logs",
    "workspace",
    "models",
    "node_modules",
    "*.zip",
    "s-open-source-export",
    "backups",
    "archive",
    "*.egg-info",
    "dist",
    "build",
    "reports",
    "*.jsonl",
    "endogenous_state.json",
    ".DS_Store",
    "heartbeat.log",
]


def should_skip(path: Path, rel: str) -> bool:
    """判断是否应排除该路径"""
    rel_lower = rel.replace("\\", "/").lower().strip("/")
    parts = rel_lower.split("/")
    if rel_lower.startswith(".env") and "example" not in rel_lower:
        return True
    if rel_lower == ".git" or rel_lower.endswith("/.git"):
        return True
    if rel_lower in (".venv", "venv") or rel_lower.endswith("/.venv") or rel_lower.endswith("/venv"):
        return True
    if "__pycache__" in parts or rel_lower.endswith(".pyc"):
        return True
    if rel_lower.endswith(".db") or ".db-shm" in rel_lower or ".db-wal" in rel_lower:
        return True
    if rel_lower == "workspace" or rel_lower.startswith("workspace/"):
        return True
    if rel_lower == "models" or rel_lower.startswith("models/"):
        return True
    if rel_lower.endswith(".zip"):
        return True
    if rel_lower == "s-open-source-export" or rel_lower.startswith("s-open-source-export/"):
        return True
    if rel_lower == "backups" or rel_lower.startswith("backups/"):
        return True
    if rel_lower == "archive" or rel_lower.startswith("archive/"):
        return True
    if ".idea/" in rel_lower or ".vscode/" in rel_lower:
        return True
    if rel_lower.endswith(".log") or rel_lower == "logs" or rel_lower.startswith("logs/"):
        return True
    if "node_modules" in parts:
        return True
    if rel_lower.endswith("endogenous_state.json") or rel_lower.endswith(".jsonl"):
        return True
    return False


def copy_tree(src: Path, dst: Path) -> None:
    """复制目录树并应用排除规则"""
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel_root = root_p.relative_to(src) if root_p != src else Path(".")
        rel_str = str(rel_root).replace("\\", "/")
        if rel_str != "." and should_skip(root_p, rel_str):
            dirs.clear()
            continue
        # 过滤子目录
        to_remove = []
        for d in dirs:
            rel = (rel_root / d) if rel_str != "." else Path(d)
            rel_s = str(rel).replace("\\", "/")
            if should_skip(root_p / d, rel_s):
                to_remove.append(d)
        for d in to_remove:
            dirs.remove(d)
        out_dir = dst / rel_root
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.startswith(".") and f != ".env.example" and f != ".gitignore":
                continue
            rel_file = (rel_root / f) if rel_str != "." else Path(f)
            rel_file_s = str(rel_file).replace("\\", "/")
            if should_skip(root_p / f, rel_file_s):
                continue
            if rel_file_s.endswith(".db") or ".db-shm" in rel_file_s or ".db-wal" in rel_file_s:
                continue
            shutil.copy2(root_p / f, out_dir / f, follow_symlinks=False)


def main():
    os.chdir(ROOT)
    date_suffix = datetime.now().strftime("%Y%m%d")
    zip_name = ROOT / f"s-open-source-{date_suffix}.zip"

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    print("正在复制项目（排除敏感与生成文件）...")
    copy_tree(ROOT, EXPORT_DIR)
    # 确保 .env.example 和 .gitignore 存在
    if (ROOT / ".env.example").exists():
        shutil.copy2(ROOT / ".env.example", EXPORT_DIR / ".env.example")
    if (ROOT / ".gitignore").exists():
        shutil.copy2(ROOT / ".gitignore", EXPORT_DIR / ".gitignore")
    print("正在打包...")
    # 压缩包内带一层目录 s-open-source-export/，解压后不会散落一堆文件
    shutil.make_archive(
        str(zip_name).replace(".zip", ""),
        "zip",
        root_dir=EXPORT_DIR.parent,
        base_dir=EXPORT_DIR.name,
    )
    print(f"已生成: {zip_name}")
    print(f"解压目录（可删除）: {EXPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
