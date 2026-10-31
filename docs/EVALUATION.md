# PolySignal - evaluation methodology

## Scientific objective

> Do social-pressure features provide incremental predictive value over
> market-implied probabilities under a no-leakage evaluation?

The implementation answers this question reproducibly. It does not assume that
PolySignal beats Polymarket.

## Branch 09 implementation

`polysignal.evaluation.backtest` implements the first conservative evaluation:
one final pre-close prediction per resolved binary market.

The CLI is:

```bash
python -m polysignal evaluate list-profiles
python -m polysignal evaluate inspect-profile --benchmark-profile real_optional
python -m polysignal evaluate build-dataset --overwrite
python -m polysignal evaluate backtest --overwrite --export-csv
python -m polysignal evaluate inspect-backtest --benchmark-profile real_optional
```

The no-flag default is `real_optional` (real resolved contracts, canonical
pre-close prices). Use `--benchmark-profile real_social_enriched_main` for the
Branch 12 public-web enriched social root. Legacy `--dataset-mode real` (and
`--dataset-mode auto`) remain compatibility aliases for `real_optional`; the
earlier synthetic benchmark pathway has been removed entirely.

Time-sliced evaluation at 7d / 3d / 1d remains deferred until price and
social-feature history are broad enough to support it cleanly.

## Evaluation dataset

The Branch 08 input is:

```text
analytical/features/social_pressure.parquet
```

Ground truth comes from the canonical Branch 07 export when it exists:

```text
analytical/evaluation/resolved_market_labels.parquet
```

When that export is absent under an isolated evaluation root, Branch 09 reuses
`polysignal.evaluation.resolved_labels.build_resolved_labels` over:

```text
normalized/polymarket/markets.parquet
normalized/polymarket/outcomes.parquet
```

The join key is `market_id`. Only binary YES/NO resolved targets enter scoring.
Repeated feature rows are deduplicated deterministically by `market_id`.

## Leakage safeguards

- Branch 08 input is rejected if it contains known target, winner, resolution,
  or expected-direction columns.
- Targets are joined only after the social feature table passes validation.
- Learned models receive an explicit social-feature whitelist.
- `real_optional` and `real_social_enriched_main` market-only probabilities are
  the latest canonical YES price points satisfying `timestamp <= close_time`.
- Settled `Outcome.current_price` values are never used as prediction inputs.
- Learned predictions are out-of-fold. If a fold cannot fit both classes, the
  model falls back to a Laplace-smoothed training base rate.

Branch 08 already enforces pre-close admissibility for social evidence. Branch
09 preserves that discipline. Both profiles consume the canonical Branch 08
feature parquet read-only; no target-informed overlay is ever generated.

## Baselines

Four baselines are reported:

1. **Constant/base-rate.** Out-of-fold Laplace-smoothed training base rate.
2. **Market-only.** Latest valid canonical pre-close YES probability
   (`timestamp <= close_time`).
3. **Social-only.** Out-of-fold regularized logistic regression over the
   explicit Branch 08 social-feature whitelist.
4. **Market + social.** The same model with `market_probability` added.

The social whitelist is intentionally small and interpretable:

```text
stance_balance
pressure_strength
temporal_decay_pressure
pressure_acceleration
evidence_quality_score
source_diversity_score
accepted_evidence_ratio
engagement_log_total
```

## Metrics and outputs

Primary metrics are Brier score and log loss. Secondary metrics are accuracy,
AUC when both classes are present, Expected Calibration Error, calibration
bins, and the confusion matrix at threshold `0.5`.

The common comparison is restricted to rows with a valid profile market
probability. Lift is:

```text
delta_brier    = market_only_brier    - market_social_brier
delta_log_loss = market_only_log_loss - market_social_log_loss
```

Positive values mean the combined model improved.

Outputs are written under:

```text
analytical/backtests/<profile>/
```

Each profile directory also contains `benchmark_profile.json` and
`warnings.json`. Dashboard-facing rows carry profile name, benchmark type,
prior source, controlled-signal flag, and claim-boundary flags.

See [EVALUATION.md](EVALUATION.md)
for the measured current run and its limitations.
