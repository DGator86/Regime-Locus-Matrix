#!/usr/bin/env python3
import json
d = json.load(open("/opt/Regime-Locus-Matrix/data/processed/universe_trade_plans.json"))
for r in d["results"]:
    if r.get("status") == "active" or r.get("decision", {}).get("action") not in (None, "skip"):
        print(json.dumps(r, indent=2))
        break
else:
    print(json.dumps(d["results"][0], indent=2))

print("\n=== ACTIVE RANKED ===")
print(json.dumps(d.get("active_ranked", [])[:3], indent=2))
