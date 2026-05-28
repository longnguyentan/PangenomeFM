# Clean Repository Structure

The clean repo keeps code directly under `src/` for easier browsing by
collaborators. A small `src/graphgenomefm.py` entry module preserves the stable
`python -m graphgenomefm` command.

The top-level `data/` directory is reserved for downloaded or derived datasets.

```text
src/
  graphgenomefm.py
  cli.py
  pipeline.py
  metadata.py

  data/
    layout.py
    gfa_parser.py
    make_benchmark.py

  graph/
    io.py
    slicing.py
    features.py
    neg_sampling.py
    qc.py

  models/
    dual_stream_gat.py
    gat.py
    heads.py
    positional_encoding.py
    attention_window.py
    losses.py

  training/
    pretrain.py

  tasks/
    ccre/
      encoding.py
      label_nodes.py
      binary.py
      baselines.py
      train_gat.py

  evaluation/
    splits.py
    lr_metrics.py

  analysis/
    network.py
    latent.py

  utils/
    versioning.py

data/
  README.md
  hprc/
    full_segments.csv
    full_links.csv

docs/
  RUN_HPRC.md
  PSB_EXPERIMENT_PLAN.md
  REPO_STRUCTURE.md

tests/
  test_data_layout.py
  test_gfa_parser.py

README.md
pyproject.toml
requirements.txt
.gitignore
CHANGELOG.md
```

After `pip install -e .`, Python packaging may create
`src/graphgenomefm.egg-info/`. That folder is generated metadata for the
editable install, is ignored by Git, and is not source code.

Do not carry over the old `scripts/` or `pgb/` directories into the final clean
repo. Their active logic has been moved into `src/`.

## Mental Model

- `data/`: how raw or cleaned graph files enter the project.
- `graph/`: graph-table operations that do not know about neural networks.
- `models/`: neural model definitions and loss/head components.
- `training/`: self-supervised foundation-model training.
- `tasks/`: downstream biological tasks.
- `evaluation/`: reusable metrics and split helpers.
- `analysis/`: paper figures and representation analysis.
- `utils/`: small project utilities.
