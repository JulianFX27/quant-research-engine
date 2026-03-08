# ADR — Anchor MR Pure 8p Research Conclusion

Date: 2026-03-07  
Project: Invariant Engine  
Strategy: Anchor Mean Reversion (8p threshold)

---

# Context

A mean reversion strategy around the New York anchor price was researched using the
`AnchorReversionAdapterBT` strategy within the Invariant backtesting engine.

Initial versions showed unstable results due to:

- volatility regime sensitivity
- execution realism assumptions
- entry timing degradation

The goal of the research sprint was to identify a configuration that remains robust
under realistic execution assumptions.

---

# Data

Instrument:
EUR/USD

Timeframe:
M5

Dataset:

EURUSD_M5_2019-01-02 → 2024-12-31

Rows:
373,632

Feature dataset:

data/anchor_reversion_fx/data/eurusd_m5_features.csv

---

# Strategy Logic

Entry condition:

Price deviates ≥ 8 pips from the New York anchor price.

Position:

Mean reversion toward anchor.

Risk:

SL = 10 pips  
TP = 15 pips  

Trade management:

Max hold:
96 bars

Daily trade cap:
1 trade per day

---

# Execution Model

Execution assumptions used in canonical benchmark:

Price mode:
synthetic_bid_ask

Spread:
1.0 pip

Slippage:
0.2 pip

Fill mode:
next_open

Intrabar path:
OHLC

Tie resolution:
SL first

---

# Regime Filter

Research showed significant degradation during high volatility regimes.

A volatility policy gate was introduced:

Policy:

VOL_HIGH blocked

Config:

configs/research/gates/volatility_gate_v1__block_vol_high.yaml

Effect:

High-volatility regimes are excluded from strategy operation.

---

# Canonical Benchmark Set

Three benchmark configurations were frozen.

---

## Primary Research Benchmark

Config:

configs/research/canonical/anchor_research_freqcap_volgate_v1.yaml

Definition:

- VOL_HIGH off
- spread 1.0
- slippage 0.2
- entry_delay = 0

Reference run:

20260307_054107_226855_ca05e0b3

Key metrics:

n_trades = 1286  
winrate = 73.87%  
expectancy_R = 0.537  
profit_factor_R = 3.086  
max_drawdown_R_abs = 13.27  

Interpretation:

Best realistic research benchmark.

---

## Operational Benchmark

Config:

configs/research/canonical/anchor_research_freqcap_volgate_operational_v1.yaml

Definition:

- VOL_HIGH off
- spread 1.0
- slippage 0.2
- entry_delay = 1

Reference run:

20260307_054251_337948_9138cf80

Key metrics:

n_trades = 1289  
winrate = 55.31%  
expectancy_R = 0.173  
profit_factor_R = 1.423  

Interpretation:

Operational degradation benchmark.

Represents realistic latency conditions.

---

## Stress Boundary Benchmark

Config:

configs/research/canonical/anchor_research_freqcap_volgate_stress_v1.yaml

Definition:

- VOL_HIGH off
- spread 1.5
- slippage 0.5
- entry_delay = 1

Reference run:

20260307_054533_882769_2aca93b5

Key metrics:

n_trades = 1289  
winrate = 52.60%  
expectancy_R = 0.044  
profit_factor_R = 1.098  

Interpretation:

Failure boundary for the strategy.

Combined execution degradation compresses the edge to near break-even.

---

# Key Findings

1. Strategy is strongly regime-dependent.

2. High volatility regimes degrade performance significantly.

3. Isolated cost increases (spread/slippage) are tolerable.

4. Timing degradation (entry delay) impacts the edge more strongly than costs.

5. Combined latency and adverse execution costs compress the edge sharply.

---

# Operational Implications

Strategy viability requires execution conditions close to:

entry_delay ≤ 1 bar

spread ≈ 1.0–1.5 pips

slippage < 1 pip

---

# Decision

The following configuration is frozen as the canonical research configuration:

Anchor MR Pure 8p  
+ Frequency cap (1 trade/day)  
+ Volatility regime filter (VOL_HIGH blocked)

Benchmark hierarchy:

1. Primary research benchmark
2. Operational benchmark
3. Stress boundary benchmark

These benchmarks will be used for future research comparisons and deployment readiness validation.

---

# Next Steps

1. Validate real execution conditions in paper / shadow pipeline.

2. Measure real slippage and latency.

3. Compare live execution to operational benchmark.

4. Evaluate session-level adversity patterns.