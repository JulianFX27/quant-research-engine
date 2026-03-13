from __future__ import annotations

from pathlib import Path
import itertools
import yaml


OUT_DIR = Path("configs/strategies/batches/ny_opening_momentum_v2_eurusd")

THRESHOLD_QS = [0.75, 0.80, 0.85]
IMPULSE_EFFS = [0.65, 0.70, 0.75]
ENTRY_DELAYS = [25, 30, 35]
HOLDINGS = [20, 30, 40]


def make_name(threshold_q: float, eff: float, entry_delay: int, holding: int) -> str:
    tq = str(threshold_q).replace(".", "")
    ef = str(eff).replace(".", "")
    return f"nyomv2_eurusd_tq{tq}_eff{ef}_ed{entry_delay}_h{holding}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for threshold_q, eff, entry_delay, holding in itertools.product(
        THRESHOLD_QS, IMPULSE_EFFS, ENTRY_DELAYS, HOLDINGS
    ):
        name = make_name(threshold_q, eff, entry_delay, holding)
        cfg = {
            "strategy_name": "NY_OPENING_MOMENTUM_V2",
            "symbol": "EURUSD",
            "data": {
                "csv_path": "data/canonical/EURUSD_M5_2014-01-01__2024-12-31__dukascopy_v2_clean.csv",
            },
            "signal": {
                "threshold_q": threshold_q,
                "impulse_efficiency_min": eff,
                "entry_delay_min": entry_delay,
                "holding_min": holding,
            },
            "execution": {
                "pip_size_price": 0.0001,
                "cost_pips": 1.0,
            },
            "outputs": {
                "out_dir": f"results/strategies/ny_opening_momentum_v2/batch_eurusd/{name}",
            },
        }

        with open(OUT_DIR / f"{name}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        count += 1

    print(f"Generated {count} configs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
