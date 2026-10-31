# PolySignal — refresh policy

The dashboard and CLI share one refresh policy that decides what work
to do when the user (or a scheduler) requests an update.

The policy is implemented in
[`src/polysignal/scheduling/refresh.py`](../src/polysignal/scheduling/refresh.py)
as a small deterministic state machine. It has no side effects: it
takes the current state plus a request, and returns the **set of steps**
that should run. Whoever calls the policy is responsible for executing
those steps.

## Refresh modes

| Mode      | Allowed steps                                                                                  | Typical trigger                           |
| --------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `FAST`    | refresh latest prices, recompute Social Pressure Index from cached features, refresh UI views | every few minutes from the dashboard       |
| `HEAVY`   | new social ingestion, LLM relevance, stance recomputation, NER, graph updates, feature rebuild | on user request, or once per N hours      |
| `MANUAL`  | whichever steps the user explicitly checks in the dashboard                                    | demo / debugging                          |

## Step catalogue

The policy operates on a fixed, ordered list of steps. Each step has a
declared cost class (`light` or `heavy`), an idempotency property, and
a clear dependency on previous steps:

```
1. fetch_latest_prices       (light)
2. ingest_social_incremental (heavy)
3. link_posts                (heavy)
4. nlp_enrich                (heavy)
5. stance_classify           (heavy)
6. graph_update              (heavy)
7. compute_social_pressure   (light if cached features, heavy otherwise)
8. compute_probability       (light)
9. refresh_dashboard_views   (light)
```

`FAST` runs `{1, 7 (cached), 8, 9}`. `HEAVY` runs all steps. `MANUAL`
runs the subset selected by the caller, respecting dependencies.

## Cooldowns

To prevent accidental over-refresh, the policy enforces minimum
intervals per step:

| Step                       | Minimum interval |
| -------------------------- | ---------------- |
| `fetch_latest_prices`      | 30 s             |
| `ingest_social_incremental`| 10 min           |
| `link_posts`               | 10 min           |
| `nlp_enrich`               | 10 min           |
| `stance_classify`          | 10 min           |
| `graph_update`             | 15 min           |
| `compute_social_pressure`  | 60 s             |
| `compute_probability`      | 30 s             |
| `refresh_dashboard_views`  | 5 s              |

The intervals are configurable. A `MANUAL` refresh can override
cooldowns by passing `force=True`.

## State

The policy keeps a per-step "last successful run" timestamp. In the
current foundation this state is held in memory; later it can be
persisted to `data/cache/refresh_state.json` without changing the
public interface.

## Determinism

Given the same `(now, last_run_table, request)` tuple, the policy
always returns the same plan. The unit tests in
`tests/test_refresh_policy.py` pin this behavior.
