import json
from pathlib import Path
import pandas as pd

RIDS = [
    "20260217_235106_310937_d30aa3ae",
    "20260217_235300_600346_8637bb1a",
    "20260217_235427_795198_31060b2c",
    "20260217_235702_952382_a7692a4a",
]

def pick(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

rows = []
for rid in RIDS:
    base_p = Path(f"results/runs/{rid}/gating_v4/mc/mc_baseline.json")
    gate_p = Path(f"results/runs/{rid}/gating_v4/mc/mc_gated_v4.json")

    row = {"rid": rid}

    if base_p.exists():
        jb = json.loads(base_p.read_text(encoding="utf-8"))
        # choose a representative MC: daily_blocks_5d is usually the strictest dependence model
        b = jb.get("mc_daily_blocks_5d", {}) or jb.get("mc_iid_trades", {})
        row.update({
            "base_totalR_p5": pick(b, "total_R_p5"),
            "base_dd_p50": pick(b, "max_dd_R_p50"),
            "base_dd_p95": pick(b, "max_dd_R_p95"),
            "base_prob_dd_ge_8": pick(b, "prob(max_dd_R>=8)"),
            "base_prob_dd_ge_12": pick(b, "prob(max_dd_R>=12)"),
        })
    else:
        row["base_status"] = "missing"

    if gate_p.exists():
        jg = json.loads(gate_p.read_text(encoding="utf-8"))
        g = jg.get("mc_daily_blocks_5d", {}) or jg.get("mc_iid_trades", {})
        row.update({
            "gate_totalR_p5": pick(g, "total_R_p5"),
            "gate_dd_p50": pick(g, "max_dd_R_p50"),
            "gate_dd_p95": pick(g, "max_dd_R_p95"),
            "gate_prob_dd_ge_8": pick(g, "prob(max_dd_R>=8)"),
            "gate_prob_dd_ge_12": pick(g, "prob(max_dd_R>=12)"),
        })
    else:
        row["gate_status"] = "missing"

    # deltas (if present)
    if "base_dd_p95" in row and "gate_dd_p95" in row and row["base_dd_p95"] is not None and row["gate_dd_p95"] is not None:
        row["d_dd_p95"] = row["gate_dd_p95"] - row["base_dd_p95"]
    if "base_prob_dd_ge_12" in row and "gate_prob_dd_ge_12" in row and row["base_prob_dd_ge_12"] is not None and row["gate_prob_dd_ge_12"] is not None:
        row["d_prob_dd_ge_12"] = row["gate_prob_dd_ge_12"] - row["base_prob_dd_ge_12"]

    rows.append(row)

df = pd.DataFrame(rows)
out = Path("results/research/gating_v4_mc_master_summary.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print("WROTE:", out)
print(df.to_string(index=False))
