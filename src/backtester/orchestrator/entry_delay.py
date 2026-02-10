# src/backtester/orchestrator/entry_delay.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from backtester.core.contracts import OrderIntent


@dataclass(frozen=True)
class EntryDelayReport:
    delay_bars: int
    attempted: int
    shifted: int
    dropped_out_of_range: int


def apply_entry_delay(
    intents_by_bar: List[List[OrderIntent]],
    *,
    delay_bars: int,
) -> Tuple[List[List[OrderIntent]], EntryDelayReport]:
    """
    Shift ALL intents forward by delay_bars (i -> i+delay_bars).

    Rationale:
      - Keeps strategy untouched (signal is identical).
      - Only changes execution timing to test "edge maturation".

    Semantics:
      - delay_bars <= 0 => returns original intents and report with zeros.
      - intents that would fall beyond last bar are dropped (counted in report).
      - preserves per-bar list order (engine will still take intents[0] today).

    NOTE:
      - This is a research transform. Do NOT use in production without explicit config.
    """
    n_bars = int(len(intents_by_bar) or 0)
    d = int(delay_bars or 0)

    if n_bars == 0:
        return intents_by_bar, EntryDelayReport(delay_bars=d, attempted=0, shifted=0, dropped_out_of_range=0)

    if d <= 0:
        attempted = sum(1 for lst in intents_by_bar if lst)
        return intents_by_bar, EntryDelayReport(delay_bars=d, attempted=attempted, shifted=0, dropped_out_of_range=0)

    out: List[List[OrderIntent]] = [[] for _ in range(n_bars)]

    attempted = 0
    shifted = 0
    dropped = 0

    for i, intents in enumerate(intents_by_bar):
        if not intents:
            continue
        attempted += len(intents)

        j = i + d
        if j >= n_bars:
            dropped += len(intents)
            continue

        out[j].extend(intents)
        shifted += len(intents)

    return out, EntryDelayReport(delay_bars=d, attempted=attempted, shifted=shifted, dropped_out_of_range=dropped)
