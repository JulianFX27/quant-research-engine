# Backtester Skeleton

Modular, reproducible backtesting framework scaffold (multi-strategy, multi-asset ready).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_single.py --config configs/run_example.yaml
```

## Design goals
- Strict data contract (normalized bars)
- Strategy plug-in interface
- Event-driven execution engine (bar-by-bar baseline)
- Explicit cost / fill models
- Reproducible run manifests
- Orchestrator for parallel experiments (to extend)

This is intentionally minimal; extend modules under `src/backtester/`.
