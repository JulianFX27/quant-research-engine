\# Freeze — PF\_FX\_NY\_UP\_EXTREME\_R1

Date: 2026-03-15



\## Status

Discovery and external validation complete. Strategy frozen for formal validation inside Invariant.



\## Canonical Strategy Name

PF\_FX\_NY\_UP\_EXTREME\_R1



\## Hypothesis

FX majors exhibit a tradable intraday mean reversion pattern after upward extreme moves during the New York open window.



\## Frozen Specification

\- Session: NEW\_YORK DST-aware

\- Session helper: `src.research.opening\_momentum.session\_times.ny\_open\_utc`

\- Session window: 150 minutes from NY open

\- Event window: 30 minutes

\- Event threshold: z >= 2.0

\- Event direction: UP\_EXTREME

\- Entry delay: 0 bars

\- Take profit: 15 pips

\- Stop loss: 12 pips

\- Time stop: 20 minutes

\- Baseline cost: 1.0 pip



\## Dataset Contract

\### EURUSD

\- `data/canonical/EURUSD\_M5\_2014-01-01\_\_2024-12-31\_\_dukascopy\_v2\_clean.csv`

\- pip\_size = 0.0001



\### GBPUSD

\- `data/canonical/GBPUSD\_M5\_2014-01-01\_\_2024-12-31\_\_dukascopy\_v2\_clean.csv`

\- pip\_size = 0.0001



\### USDJPY

\- `data/canonical/USDJPY\_M5\_2014-01-01\_\_2024-12-31\_\_dukascopy\_v2\_clean.csv`

\- pip\_size = 0.01



\## Validation Window

2019-01-02 to 2024-12-31



\## Results Summary



\### EURUSD

\- trades: 560

\- winrate: 55.36%

\- expectancy\_pips: +1.139

\- avg\_win\_pips: +7.33

\- avg\_loss\_pips: -6.54

\- profit\_factor: 1.390



\### GBPUSD

\- trades: 574

\- winrate: 55.23%

\- expectancy\_pips: +1.203

\- avg\_win\_pips: +8.71

\- avg\_loss\_pips: -8.06

\- profit\_factor: 1.333



\### USDJPY

\- trades: 559

\- winrate: 48.48%

\- expectancy\_pips: +0.423

\- avg\_win\_pips: +7.71

\- avg\_loss\_pips: -6.43

\- profit\_factor: 1.127



\## Interpretation

\- Strategy validated on canonical datasets and DST-aware session logic.

\- EURUSD and GBPUSD form the core tradable universe.

\- USDJPY remains positive but weaker; treat as secondary or lower-confidence asset.

\- No further parameter optimization should be done outside Invariant.

\- Next stage belongs to formal engine validation:

&#x20; - temporal splits

&#x20; - walk-forward

&#x20; - portfolio simulation

&#x20; - Monte Carlo

&#x20; - prop-firm rules



\## Frozen Core Universe

\- EURUSD

\- GBPUSD



\## Frozen Secondary Universe

\- USDJPY



\## Immediate Next Step

Port the frozen strategy into Invariant for formal validation and portfolio-level testing.



