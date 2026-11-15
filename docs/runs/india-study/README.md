# India study evidence subset

These files are copied or derived from the completed local `india-study` run
reviewed on 15 November 2026. They contain no raw market data, model weights,
testing files or absolute host paths.

`aggregate_metrics.csv` exactly copies the generated summary.
`unit_metrics_5bps.csv` and `unit_regimes_5bps.csv` select `cost_bps=5` from
observed per-unit tables, preserving numeric text. Repeated deterministic
baseline rows remain for traceability, not as independent observations.
The manifest and identity JSON files are exact copies.

See the [study review](../../analysis-2026-11-15.md) for definitions and limits.
