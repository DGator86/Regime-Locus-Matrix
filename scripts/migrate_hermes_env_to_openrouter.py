#!/usr/bin/env python3
"""Point Hermes env vars at OpenRouter using OPENROUTER_API_KEY (VPS/local .env).

Comments prior RLM_HERMES_* model/URL/API lines so Groq or other hosts do not override.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _parse_val(text: str, key: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(key + "="):
            val = s.split("=", 1)[1].strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            return val
    return None


def main() -> int:
    p = Path("/opt/Regime-Locus-Matrix/.env")
    if not p.is_file():
        print("error: expected", p, file=__import__("sys").stderr)
        return 2
    text = p.read_text(encoding="utf-8")
    or_key = _parse_val(text, "OPENROUTER_API_KEY")
    if not or_key:
        print("error: OPENROUTER_API_KEY missing in .env", file=__import__("sys").stderr)
        return 1

    prefixes = (
        "RLM_HERMES_BASE_URL=",
        "RLM_HERMES_API_KEY=",
        "RLM_HERMES_MODEL=",
        "RLM_HERMES_OPENROUTER_MODEL=",
        "RLM_HERMES_FALLBACK_MODEL=",
        "RLM_HERMES_FALLBACK_BASE_URL=",
        "RLM_HERMES_FALLBACK_API_KEY=",
    )

    out_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            out_lines.append(line)
            continue
        hit = False
        for pref in prefixes:
            if s.startswith(pref):
                out_lines.append("# " + s + "  # commented: Hermes migrated to OpenRouter")
                hit = True
                break
        if not hit:
            out_lines.append(line)

    body = "\n".join(out_lines).rstrip()
    marker = "\n# Hermes → OpenRouter (managed)"
    if marker.strip() in body:
        body = body.split(marker)[0].rstrip()

    body = body + marker + "\n"
    body += "RLM_HERMES_BASE_URL=https://openrouter.ai/api/v1\n"
    body += f"RLM_HERMES_API_KEY={or_key}\n"
    body += "RLM_HERMES_MODEL=meta-llama/llama-3.2-3b-instruct:free\n"

    backup = p.parent / f".env.bak.hermes-openrouter.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    backup.write_text(text, encoding="utf-8")
    p.write_text(body + "\n", encoding="utf-8")
    print("ok:", p, "backup:", backup.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
