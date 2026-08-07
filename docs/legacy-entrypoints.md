# Historical entry-point migration

The research repository accumulated one root script per backend, model, metric and trial.
They were intentionally excluded from the clean public tree after their prompt variants
and evaluation families were consolidated into `llm_x.cli`, `llm_x.evaluation`,
`llm_x.backends` and `llm_x.metrics`.

## Consolidation map

| Historical family | Public replacement |
|---|---|
| `main.py`, `main_beta_version.py`, `main_llama3*.py`, `main_vicuna*.py`, `main_ollama_llama3_70b_q8*.py` | `--metric next-action` or `last-action`, with an explicit backend and question name |
| `main_continuous*.py`, continuous Ollama variants | `--metric next-action` with a registered continuous bins/no-bins question |
| `main_last_action_continuous*.py` | `--metric last-action` with a registered continuous question |
| `main_next_state*.py`, `main_last_state*.py` | `--metric next-state` or `last-state` |
| `main_argue_action*.py` | `--metric argue-action`; continuous argument trials are not supported in public v1 |
| `main_minigrid_beta_version.py` | Registered task/prompt support remains; validate the high-dimensional trajectory before use |
| `main_random.py` and `*_random.py` | Excluded research baselines; deterministic argument sampling is implemented centrally |
| `*_trial*.py`, `main_llama3_REMOVED.py` | Excluded incomplete experiments |

## Inventoried source scripts

The migration review covered the following 47 root entry points:

```text
main.py
main_argue_action.py
main_argue_action_beta_version.py
main_argue_action_continuous_beta_version.py
main_beta_version.py
main_continuous.py
main_continuous_beta_version.py
main_continuous_bins.py
main_continuous_bins_fetch.py
main_continuous_bins_random.py
main_continuous_no_bins.py
main_continuous_no_bins_fetch.py
main_continuous_no_bins_random.py
main_continuous_random.py
main_last_action.py
main_last_action_beta_version.py
main_last_action_continuous_beta_version.py
main_last_state.py
main_last_state_beta_version.py
main_last_state_fetch.py
main_llama3.py
main_llama3_REMOVED.py
main_llama3_beta_version.py
main_llama3_last_action.py
main_llama3_last_action_beta_version.py
main_llama3_next_state.py
main_minigrid_beta_version.py
main_next_state.py
main_next_state_beta_version.py
main_next_state_discrete.py
main_next_state_random.py
main_next_state_trial.py
main_next_state_trial_fetch.py
main_next_state_trial_fetch_part.py
main_ollama_llama3_70b_q8.py
main_ollama_llama3_70b_q8_argue_action_beta_version.py
main_ollama_llama3_70b_q8_beta_version.py
main_ollama_llama3_70b_q8_continuous_beta_version.py
main_ollama_llama3_70b_q8_last_action_beta_version.py
main_ollama_llama3_70b_q8_last_action_continuous_beta_version.py
main_ollama_llama3_70b_q8_last_state_beta_version.py
main_ollama_llama3_70b_q8_next_state_beta_version.py
main_random.py
main_vicuna.py
main_vicuna_beta_version.py
main_vicuna_last_action.py
main_vicuna_next_state.py
```

The original research repository remains the archival reference for these files. They are
not required to operate the public evaluator.
