# Architecture

This scaffold is organized as:
- `backtester/data`: ingestion + normalization
- `backtester/strategies`: strategy plugins
- `backtester/execution`: fill/cost models + engine
- `backtester/metrics`: stats and diagnostics
- `backtester/orchestrator`: experiment runners

Extend toward:
- multi-position, portfolio allocator
- walk-forward and monte-carlo modules
- replay/live adapters
- instrument specs + calendars
