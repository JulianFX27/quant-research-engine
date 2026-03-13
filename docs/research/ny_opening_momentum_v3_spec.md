# NY Opening Momentum v3 — Research Specification

Status: ACTIVE (Research Phase)

Date: 2026-03-13  
Project: Invariant Engine  
Branch: live_shadow_v1


---

# 1. Research Objective

NY Opening Momentum v3 is the next iteration of the NY opening momentum research family.

The purpose of v3 is to test whether the signal observed in NY Opening Momentum v2 becomes viable when restricted to a specific impulse-quality regime.

NYOM v2 failed as an unconditional standalone strategy in walk-forward validation, but post-OOS regime diagnostics showed evidence of conditional behavior, especially in GBPUSD.

Therefore, v3 is designed to test a narrower and more causal hypothesis.


---

# 2. Core Hypothesis

The NY opening momentum effect persists only when the first 30-minute NY impulse is:

- materially strong
- directionally efficient
- not extreme enough to imply short-term exhaustion

In other words, continuation is expected in strong-but-not-extreme directional impulses.


---

# 3. Research Design Principle

v3 is not intended as a retrospective "best bucket selection" strategy.

Instead, it is a pre-defined ex ante test of a causal interpretation derived from v2 diagnostics.

The regime gate is treated as a signal-quality filter, not as a broad optimization surface.

This means:

- the impulse is first observed
- its strength and efficiency are measured
- the trade is taken only if the event belongs to the predefined operable regime


---

# 4. Event Definition

The event is defined from the first 30 minutes of the New York session.

Variables:

- ret_30m = directional return from NY open to NY open + 30m
- range_30m = high-low range over the same window
- impulse_efficiency = abs(ret_30m) / range_30m
- direction = sign(ret_30m)

These variables define the opening impulse and its quality.


---

# 5. Execution Model

Execution remains intentionally identical to NYOM v2.

Reason:

To isolate the effect of regime gating and avoid changing multiple dimensions at once.

Execution logic:

- entry_delay = 30 minutes
- enter in the direction of ret_30m
- holding time = 30 minutes
- max_holding_bars = 6
- no TP/SL price exits
- allow_missing_sl = true
- risk_proxy_price used only for R-space metrics

This is a time-stop faithful implementation.


---

# 6. Regime Gate Policies

Two pre-defined policies are allowed in this phase.

## Policy A — Strict

- ret30_bucket = Q4
- efficiency_bucket = Q4

Interpretation:

Only trade impulses that are strong and clean, but not extreme.


## Policy B — Broader

- ret30_bucket in {Q3, Q4}
- efficiency_bucket = Q4

Interpretation:

Allows a wider strength band while preserving directional cleanliness.


---

# 7. Experimental Constraints

To reduce overfitting risk, the following constraints are imposed:

- No additional policy variants in this phase
- No optimization across Q2/Q3/Q4/Q5 grids
- No side-specific optimization
- No weekday-specific optimization
- No holding-period changes
- No TP/SL changes
- No additional volatility or session-context filters

The only new dimension introduced in v3 is non-extreme impulse-quality gating.


---

# 8. Primary Research Asset

Primary asset for v3 research:

GBPUSD

Reason:

v2 regime diagnostics showed the clearest conditional edge in GBPUSD, specifically around intermediate-to-high impulse strength and high efficiency.

Secondary replication assets:

- EURUSD
- USDJPY


---

# 9. Primary Research Question

Does NY opening momentum become a viable conditional alpha when restricted to strong-but-not-extreme and directionally efficient opening impulses?


---

# 10. Success Criteria

v3 will be considered promising only if it shows:

- improved expectancy versus v2 baseline
- acceptable trade count
- better local stability
- evidence that the conditional rule survives walk-forward validation

Research progression order:

1. research script validation
2. local robustness review
3. backtester integration
4. walk-forward validation
5. portfolio-level review


---

# 11. Failure Criteria

v3 will be rejected if:

- edge disappears after ex ante policy fixing
- trade count becomes too low to be operationally meaningful
- performance improvement is explained only by one short subperiod
- walk-forward still fails without meaningful stability improvement


---

# 12. Current Status

NY Opening Momentum v3 is now the active research successor to NY Opening Momentum v2.

NYOM v2 remains formally closed as:

- unconditional walk-forward failed
- conditional behavior detected
- successor hypothesis promoted to v3