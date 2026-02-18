import json
from pathlib import Path
import pandas as pd

RIDS = [
    "20260217_235106_310937_d30aa3ae",
    "20260217_235300_600346_8637bb1a",
    "20260217_235427_795198_31060b2c",
    "20260217_235702_952382_a7692a4a",
]

rows = []
for rid in RIDS:
    p = Path(f"results/runs/{rid}/gating_v4/gating_v4_compare.json")
    if not p.exists():
        rows.append({"rid": rid, "status": "missing_gating_v4_compare_json"})
        continue

    j = json.loads(p.read_text(encoding="utf-8"))
    b = j["baseline"]
    g = j["gated"]

    rows.append({
        "rid": rid,
        "n_base": b["n_trades"],
        "expR_base": b["expectancy_R"],
        "dd_base": b["max_dd_R"],
        "ls_base": b["max_losing_streak"],
        "n_gated": g["n_trades"],
        "expR_gated": g["expectancy_R"],
        "dd_gated": g["max_dd_R"],
        "ls_gated": g["max_losing_streak"],
        "d_expR": g["expectancy_R"] - b["expectancy_R"],
        "d_dd": g["max_dd_R"] - b["max_dd_R"],
        "d_ls": g["max_losing_streak"] - b["max_losing_streak"],
        "status": "OK",
    })

df = pd.DataFrame(rows)
out = Path("results/research/gating_v4_master_summary.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print("WROTE:", out)
print(df.to_string(index=False))
