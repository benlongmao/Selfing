#!/usr/bin/env python3
"""
[2026-02-05] Workspace path diagnostics (manual harness, not ``unittest``).

- Detect layout violations under the current workspace root
- Exercise ``normalize_path`` / ``get_standard_path_for_action``
- Print migration hints

Run from repo root:
    .venv/bin/python backend/test_workspace_paths.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.workspace_path_manager import (
    detect_workspace_violations,
    normalize_path,
    get_standard_path_for_action,
    STANDARD_DIRECTORIES,
    LEGACY_DIRECTORY_MAPPING
)


def test_path_normalization():
    """Print ``normalize_path`` outcomes for a fixed case matrix."""
    print("\n" + "=" * 80)
    print("Suite 1: path normalization")
    print("=" * 80)

    test_cases = [
        ("workspace/sandbox/autonomous_diaries/test.md", None, "diaries"),
        ("sandbox/autonomous_searches/research.md", None, "research"),
        ("workspace/sandbox/autonomous_learning/learn.md", None, "research"),
        ("diary_20260205_1530.md", "diary", "diaries"),
        ("experiment_test.md", None, "experiments"),
        ("sandbox/sandbox/test.md", None, None),
        ("sandbox/temp/draft.md", None, "drafts"),
        ("search_20260205.md", "search", "research"),
    ]

    for input_path, hint, expected_dir in test_cases:
        print(f"\n[case] input: {input_path}")
        if hint:
            print(f"  hint: {hint}")

        try:
            abs_path, rel_path, was_corrected = normalize_path(input_path, hint)
            print(f"  absolute: {abs_path}")
            print(f"  relative: {rel_path}")
            print(f"  corrected: {'yes' if was_corrected else 'no'}")

            if expected_dir:
                if expected_dir in rel_path:
                    print(f"  OK — maps under {expected_dir}/")
                else:
                    print(f"  WARN — expected segment {expected_dir}/ missing")

        except Exception as e:
            print(f"  ERROR: {e}")


def test_action_paths():
    """Print canonical paths returned for autonomous action kinds."""
    print("\n" + "=" * 80)
    print("Suite 2: standard paths per action type")
    print("=" * 80)

    action_types = [
        "diary",
        "search",
        "learning",
        "experiment",
        "analysis",
    ]

    for action_type in action_types:
        print(f"\n[action] {action_type}")
        path = get_standard_path_for_action(action_type)
        print(f"  standard path: {path}")

        if "sandbox/" in path:
            parts = path.split('/')
            if len(parts) >= 3:
                dir_name = parts[1]
                print(f"  target bucket: {dir_name}/")

                is_standard = dir_name in [
                    d["path"].split('/')[-1]
                    for d in STANDARD_DIRECTORIES.values()
                ]

                if is_standard:
                    print("  OK — bucket is whitelisted")
                else:
                    print("  FAIL — bucket not in STANDARD_DIRECTORIES")


def diagnose_workspace():
    """Summarize ``detect_workspace_violations`` for the live tree."""
    print("\n" + "=" * 80)
    print("Diagnosis: workspace violations")
    print("=" * 80)

    violations = detect_workspace_violations()

    print(f"\nTotal directories seen: {violations['total_directories']}")
    print(f"Standard directory slots: {len(STANDARD_DIRECTORIES)}")

    if violations['total_directories'] > len(STANDARD_DIRECTORIES):
        print(f"WARN — {violations['total_directories'] - len(STANDARD_DIRECTORIES)} extra dirs vs template")
    else:
        print("OK — directory count within template")

    if violations['legacy_directories']:
        print(f"\nFAIL — {len(violations['legacy_directories'])} legacy / non-standard dirs (showing up to 10):")
        for vio in violations['legacy_directories'][:10]:
            path = vio['path']
            migration_target = vio['should_migrate_to']
            vio_type = vio['type']

            if vio_type == "known_legacy":
                print(f"  - {path}/ → migrate toward {migration_target}/")
            else:
                print(f"  - {path}/ → unknown bucket, review manually")

        if len(violations['legacy_directories']) > 10:
            print(f"  ... {len(violations['legacy_directories']) - 10} more")
    else:
        print("\nOK — no legacy directories flagged")

    if violations['root_files']:
        print(f"\nWARN — {len(violations['root_files'])} loose files at workspace root:")
        for vio in violations['root_files'][:10]:
            filename = vio['filename']
            suggested_dir = vio['suggested_dir']
            print(f"  - {filename} → suggest {suggested_dir}/")

        if len(violations['root_files']) > 10:
            print(f"  ... {len(violations['root_files']) - 10} more")
    else:
        print("\nOK — sandbox root has no stray files")


def show_standard_directories():
    """Pretty-print the canonical sandbox layout."""
    print("\n" + "=" * 80)
    print("Reference layout (policy v1.3)")
    print("=" * 80)

    print("\nworkspace/sandbox/")
    print("├── 工作空间规章制度.md  (only markdown allowed at sandbox root)")

    for dir_key, dir_info in STANDARD_DIRECTORIES.items():
        dir_name = dir_info["path"].split('/')[-1]
        desc = dir_info["description"]
        print(f"├── {dir_name}/  # {desc}")

    print("\n" + "=" * 80)
    print("Legacy → canonical directory map")
    print("=" * 80)

    for old_dir, new_dir in LEGACY_DIRECTORY_MAPPING.items():
        print(f"  {old_dir}/ → {new_dir}/")


def main():
    """Run diagnostics + normalization suites."""
    print("\n" + "=" * 100)
    print("S-main — workspace path diagnostic harness")
    print("=" * 100)

    try:
        show_standard_directories()
        diagnose_workspace()
        test_path_normalization()
        test_action_paths()

        print("\n" + "=" * 100)
        print("Done")
        print("=" * 100)

        print("\nWhat changed (historical summary):")
        print("1. Central path manager lives in workspace_path_manager.py")
        print("2. Eleven canonical sandbox buckets are enumerated in STANDARD_DIRECTORIES")
        print("3. normalize_path() steers stray files into the right bucket")
        print("4. detect_workspace_violations() surfaces legacy dirs + root clutter")
        print("5. autonomous_action_engine no longer hardcodes obsolete paths")

        print("\nNext steps:")
        print("- Run the app and confirm new artifacts land in the expected buckets")
        print("- Watch logs for [PATH-NORM] to verify auto-correction")
        print("- Optionally run a one-off migration script to move legacy folders")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
