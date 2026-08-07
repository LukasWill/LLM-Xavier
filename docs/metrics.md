# Metric and indexing contract

This document distinguishes the public result schema from the historical script labels.
All rates are JSON numbers rather than rounded percentage strings.

## Query schedule

For a configured history size `H`, action and argument queries receive `H + 1` complete
historical records and query the following action. Next-state queries include the current
state-action-reward tuple in the history and score the transition to the next state.
Last-action and last-state queries require an additional following state.

Each prediction records `index`, `history_start` and `history_end_exclusive`, making the
schedule independently auditable. No task-specific truncation occurs unless `max_steps`
is explicitly configured.

## Parsing

- Discrete action: `Final action choice: [N]`.
- Continuous action: `predictions = [number, ...]`.
- State direction: `predictions = ["INC", "DEC", ...]`, with `"UNCH"` permitted by
  `more_options` prompt variants.
- Argument vote: `final_vote = [True|False]` or `Final vote is [True|False]`.

Parsing uses regular expressions and `ast.literal_eval`; arbitrary model text is never
executed. A malformed response is marked `ignored`, and its raw text is discarded.

## Action metrics

Discrete actions require exact integer equality. Continuous action components are mapped
to ten bins using the historical `numpy.digitize(..., right=True)` convention. The
default range is `[-2, 2]`, or `[-1, 1]` for Fetch tasks; it can be overridden explicitly.
A query is an exact match only when every action component has the correct bin.

## State metrics

States are rounded to five decimal places. Each component is labeled `INC` or `DEC`; a
`more_options` question additionally labels `UNCH` when the absolute change is below the
configured threshold, default `1e-4`. Fetch state vectors omit their final feature by
default for compatibility with the research evaluation.

## Argument metrics

A seed-deterministic, balanced schedule presents either the recorded discrete action or a
different recorded action. Accuracy measures whether the model's vote agrees with the
presented action's correctness. The output also preserves the paper-style argue-for rate
and the four correct/wrong by for/against counts.

## Reported aggregates

- `parse_rate`: parsed queries divided by all queries.
- `exact_match_rate_all_queries`: exact matches divided by all queries.
- `exact_match_rate_parsed_queries`: exact matches divided by parsed queries.
- `mean_element_accuracy_parsed_queries`: mean per-query element accuracy when applicable.
- `legacy_compatible_match_rate`: action families use all queries; state and argument
  families use parsed queries, matching the dominant historical denominator convention.

The two explicit exact-match rates are authoritative; the legacy-compatible value exists
only to compare with historical outputs.
