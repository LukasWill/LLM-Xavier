# LLM-X

LLM-X evaluates how well language models form a mental model of a trained
reinforcement-learning agent from its recorded state-action-reward history. This public
release provides a validated, offline-only evaluator for the action prediction, state
direction prediction and action-argument tasks used in the accompanying paper.

Paper: [Mental Modelling of Reinforcement Learning Agents by Language Models](https://openreview.net/forum?id=JN7iNWaPTe), Transactions on Machine Learning Research, 2024.

## Scope

The supported workflow consumes an existing NPZ trajectory and queries a configured chat
backend. It does not train an RL policy, execute a live Gym environment or load policy
checkpoints. Full trajectories and policy artifacts are distributed separately through
[lerrhoo/llmx_data](https://huggingface.co/datasets/lerrhoo/llmx_data).

Supported evaluation families are:

- next and last action prediction, for discrete or continuous actions;
- next and last state-direction prediction; and
- arguing for or against a presented discrete action.

Run `llm-x list-questions` for the complete model/prompt variant registry.

## Installation

LLM-X requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

For OpenAI or OpenAI-compatible APIs:

```bash
python -m pip install -e '.[openai,test]'
```

Credentials are read from an environment variable and are never accepted as a CLI value.

## Offline smoke test

The fixture backend performs no network request and stores no raw model response:

```bash
python examples/create_smoke_episode.py --output /tmp/llmx-smoke.npz
llm-x inspect-data --data-path /tmp/llmx-smoke.npz
llm-x evaluate \
  --data-path /tmp/llmx-smoke.npz \
  --task-name MountainCar-v0 \
  --metric next-action \
  --question-name next_action_prediction \
  --history-size 0 \
  --backend fixture \
  --fixture-responses examples/fixture_responses.json \
  --output-dir outputs/smoke
```

The output directory contains:

- `run.json`: versions, backend/model identity and input checksum;
- `config.effective.json`: scientifically relevant effective settings;
- `metrics.json`: numeric aggregate metrics and parser coverage; and
- `predictions.jsonl`: per-query parsed predictions, ground truth and optional prompts.

Raw LLM responses are never written. Existing output files are not replaced unless
`--overwrite` is provided.

## Real backend example

```bash
export OPENAI_API_KEY='set-this-outside-the-command-line'
llm-x evaluate \
  --data-path /path/to/episode.npz \
  --task-name MountainCar-v0 \
  --metric next-action \
  --question-name next_action_prediction \
  --history-size 3 \
  --backend openai \
  --model YOUR_MODEL_NAME \
  --max-queries 5 \
  --output-dir outputs/mountain-car
```

OpenAI-compatible and Ollama backends require an explicit `--endpoint`. Use `--dry-run`
to validate task, question and episode configuration without constructing a backend or
making a request. A YAML example is available at `configs/eval.example.yaml`; CLI values
override values loaded with `--config`.

The OpenAI backend intentionally uses the supported
[Chat Completions API](https://developers.openai.com/api/reference/chat-completions/overview)
to preserve the paper's system/user-message structure and interoperate with local servers.
OpenAI recommends the Responses API for new applications that need newer platform features;
LLM-X does not require those features. API keys follow the official
[environment-variable guidance](https://developers.openai.com/api/docs/libraries).

## Reproducibility choices

- Query indices and history ranges are recorded per prediction.
- Argument actions are balanced and selected by a recorded seed.
- State directions use a configurable threshold, defaulting to `1e-4`.
- Fetch state evaluation preserves the historical final-feature exclusion by default; use
  `--no-fetch-drop-last-feature` to disable it.
- Continuous actions use ten bins and the historical right-inclusive edge convention.
- Results report exact-match rates over all queries and over successfully parsed queries,
  avoiding the denominator ambiguity in the research scripts.
- No task is silently truncated. Use `--max-steps 50` when reproducing experiments that
  intentionally evaluated only the first 50 steps.

See [docs/metrics.md](docs/metrics.md) for definitions and
[docs/legacy-entrypoints.md](docs/legacy-entrypoints.md) for the migration from the
historical `main_*.py` scripts.

## Data, privacy and licensing

See [DATASET.md](DATASET.md) for the NPZ schema, data license and provenance policy.
Generated results, paper plots, local training runs, raw model responses and analysis
artifacts are ignored by Git. Before publishing additional assets, run:

```bash
python tools/check_release.py
```

The source code is licensed under Apache-2.0. The separately distributed dataset and
checkpoints are released under CC BY 4.0. Third-party software retains its own license.

## Citation

```bibtex
@article{lu2024mental,
  title   = {Mental Modelling of Reinforcement Learning Agents by Language Models},
  author  = {Lu, Wenhao and Zhao, Xufeng and Spisak, Josua and
             Lee, Jae Hee and Wermter, Stefan},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  url     = {https://openreview.net/forum?id=JN7iNWaPTe}
}
```
