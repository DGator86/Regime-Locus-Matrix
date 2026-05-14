#!/usr/bin/env python3
"""Print plain-English explanations for option plans (universe JSON or a single plan dict).

Examples::

    python scripts/translate_options_plain_language.py
    python scripts/translate_options_plain_language.py --plans data/processed/universe_trade_plans.json
    python scripts/translate_options_plain_language.py --plan-id AMZN_20260513_1605 --plan-id META_20260513_1605
    python scripts/translate_options_plain_language.py --all-statuses
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.notify.options_plain_language import explain_trade_plan  # noqa: E402


def _load_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return raw


def _iter_plan_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ("results", "active_ranked"):
        for r in data.get(key) or []:
            if isinstance(r, dict):
                rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--plans",
        type=Path,
        default=Path(os.environ.get("RLM_PLANS_JSON", "data/processed/universe_trade_plans.json")),
        help="Universe trade plans JSON (default: data/processed/universe_trade_plans.json)",
    )
    ap.add_argument("--plan-id", dest="plan_ids", action="append", default=[], help="Only these plan_id values (repeatable)")
    ap.add_argument(
        "--all-statuses",
        action="store_true",
        help="Include skipped/error/trimmed rows (default: active and rows matching --plan-id)",
    )
    ap.add_argument(
        "--single-plan-json",
        type=Path,
        default=None,
        help="One plan object as JSON file (overrides --plans)",
    )
    args = ap.parse_args()

    if args.single_plan_json is not None:
        plan = _load_json(args.single_plan_json)
        sys.stdout.write(explain_trade_plan(plan))
        return 0

    if not args.plans.is_file():
        print(f"[error] missing {args.plans}", file=sys.stderr)
        return 2

    data = _load_json(args.plans)
    wanted = {str(x).strip() for x in args.plan_ids if str(x).strip()}
    blocks: list[str] = []
    for row in _iter_plan_rows(data):
        pid = str(row.get("plan_id") or "")
        st = str(row.get("status") or "")
        if wanted:
            if pid not in wanted:
                continue
        elif not args.all_statuses and st and st != "active":
            continue
        blocks.append(explain_trade_plan(row))

    if not blocks:
        print("[info] no matching plan rows", file=sys.stderr)
        return 1

    sep = "\n" + ("—" * 72) + "\n\n"
    sys.stdout.write(sep.join(blocks))
    if not blocks[-1].endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
