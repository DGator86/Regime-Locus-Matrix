#!/usr/bin/env python3
"""Quick inspection of VPS data files for dashboard API design."""
import json, csv, os

ROOT = os.environ.get("RLM_DATA_DIR", "/opt/Regime-Locus-Matrix/data/processed")

def show_json(path, max_keys=3):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        keys = list(d.keys())
        print(f"  keys({len(keys)}): {keys[:10]}")
        for k in keys[:max_keys]:
            v = d[k]
            if isinstance(v, dict):
                print(f"  [{k}] sub-keys: {list(v.keys())[:15]}")
            elif isinstance(v, list) and v:
                print(f"  [{k}] list[{len(v)}], first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
            else:
                print(f"  [{k}] = {json.dumps(v)[:200]}")
    elif isinstance(d, list):
        print(f"  list[{len(d)}]")
        if d and isinstance(d[0], dict):
            print(f"  first item keys: {list(d[0].keys())[:15]}")

def show_csv(path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = sum(1 for _ in reader)
    print(f"  cols({len(header)}): {header[:15]}...")
    print(f"  rows: {rows}")

for name in sorted(os.listdir(ROOT)):
    fp = os.path.join(ROOT, name)
    print(f"\n=== {name} ===")
    if name.endswith(".json"):
        show_json(fp)
    elif name.endswith(".csv"):
        show_csv(fp)
