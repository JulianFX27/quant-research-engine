from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class RobustnessManifest:
    test_type: str
    robustness_id: str
    created_at_utc: str
    source_run_dir: str
    source_trades_csv: str
    output_dir: str
    dataset_id: Optional[str]
    dataset_fp8: Optional[str]
    run_id: Optional[str]
    method: str
    n_paths: int
    block_days: Optional[int]
    seed: int
    r_col: str
    entry_time_col: Optional[str]
    notes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessSummary:
    test_type: str
    robustness_id: str
    method: str
    n_paths: int
    n_observations: int
    p05_final_equity: float
    p50_final_equity: float
    p95_final_equity: float
    p05_max_drawdown_abs: float
    p50_max_drawdown_abs: float
    p95_max_drawdown_abs: float
    p05_max_drawdown_pct: float
    p50_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    p50_max_consecutive_losses: float
    p90_max_consecutive_losses: float
    p95_max_consecutive_losses: float
    p99_max_consecutive_losses: float
    p05_expectancy_R: float
    p50_expectancy_R: float
    p95_expectancy_R: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)