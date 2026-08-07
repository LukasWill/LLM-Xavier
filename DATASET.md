# Dataset and checkpoints

The full offline trajectories and trained-policy artifacts are maintained in the public
[lerrhoo/llmx_data Hugging Face dataset](https://huggingface.co/datasets/lerrhoo/llmx_data).
They are intentionally not duplicated in this source repository.

The dataset is released by its owner under CC BY 4.0. The Hugging Face dataset card
should carry the same license identifier (`cc-by-4.0`) and cite the accompanying paper.

## Episode contract

An episode is a NumPy `.npz` archive loaded with `allow_pickle=False`. It must contain
numeric arrays named:

- `states`, with the timestep on axis 0;
- `actions`, with the same number of timesteps;
- `rewards`, with the same number of timesteps; and
- `episodic_return`.

Fetch episodes may additionally provide `achieved_goals` and `desired_goals`. MiniGrid
episodes may provide `agent_dirs` and `dir_vectors`. Extra numeric arrays are retained
by the loader but are not implicitly inserted into prompts.

Use the explicit validator before running a paid evaluation:

```bash
llm-x inspect-data --data-path /path/to/episode.npz
```

## Privacy and provenance

Before publishing additional dataset revisions, scan filenames, NPZ keys, tensor/event
metadata and archive contents for usernames, home-directory paths, hostnames, endpoints,
credentials and raw model responses. TensorBoard event logs are generated artifacts and
are not part of the public data contract.
