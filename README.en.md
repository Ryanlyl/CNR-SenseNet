[![README Chinese](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-0F766E?style=for-the-badge)](./README.md)
[![README English](https://img.shields.io/badge/README-English-1D4ED8?style=for-the-badge)](./README.en.md)

# CNR-SenseNet

`CNR-SenseNet` is a binary spectrum sensing research repository built on top of `RML2016.10a`. It provides one consistent workflow for `signal vs noise` dataset preparation, classical baselines, deep models, paper-oriented experiments, and GPU cluster reproduction.

## Highlights

- Unified data pipeline: rebuilds and caches a `signal vs noise` dataset from `RML2016.10a_dict.pkl`, with optional `SNR` and modulation filtering.
- Unified detector interface: classical detectors and neural models follow the same `fit / predict_scores / predict` workflow.
- End-to-end experiment coverage: training, checkpoint reevaluation, hyperparameter search, model comparison, explainability, ablation, robustness, and Slurm jobs are already included.
- Reproducible artifacts: the repository already contains example plots, JSON summaries, and checkpoints under `project/plots/` and `project/results/`.

## CNR-SenseNet Architecture

The main model in `project/CNR_SenseNet.py` uses three feature branches, and each branch can be toggled for ablation studies:

- `Raw Branch`: extracts temporal features directly from the IQ sequence.
- `Energy Branch`: learns from local energy-window statistics.
- `Aux Branch`: switchable between `Diff` and `Autocorr` to model differential or autocorrelation cues.

## Repository Layout

```text
CNR-SenseNet/
|- project/
|  |- data/                     # raw data, cached archives, data builders, visualization
|  |- models/                   # Energy / Autocorr / MLP / CNN1D / LSTM and helpers
|  |- CNR_SenseNet.py           # main model implementation
|  |- prepare_dataset.py        # build or reuse cached dataset
|  |- train.py                  # train baseline models
|  |- run_cnr_sensenet_eval.py  # train and evaluate CNR-SenseNet
|  |- evaluate.py               # reevaluate saved checkpoints
|  |- search_cnr_sensenet.py    # grid search
|  |- run_model_comparison.py   # unified multi-model comparison
|  |- explainability.py         # explainability analysis
|  |- ablation.py               # branch ablation studies
|  |- robustness.py             # robustness evaluation
|  |- plots/                    # generated figures
|  `- results/                  # search, ablation, robustness, and other outputs
`- cluster/
   |- environment.yml           # Conda environment definition
   |- setup_env.sh              # cluster environment bootstrap
   |- jobs/*.sbatch             # Slurm job templates
   `- README.md                 # cluster runbook
```

## Quick Start

### 1. Create the environment

The simplest option is to reuse `cluster/environment.yml`:

```bash
conda env create -f cluster/environment.yml
conda activate cnr-sensenet
```

Notes:

- This environment file includes `pytorch-cuda=12.1`, so it is best suited for GPU or cluster setups.
- For CPU-only local machines, keep the same dependency set but install the appropriate PyTorch build manually.

### 2. Prepare the data

Place the raw dataset at:

```text
project/data/RML2016.10a_dict.pkl
```

Then build or reuse the cached archive:

```bash
python -m project.prepare_dataset
```

If you do not have the raw `.pkl` yet, you can still work with the cached archives already stored under `project/data/processed/` by passing `--dataset-npz` to the training scripts.

### 3. Train baseline models

```bash
python -m project.train \
  --models energy_detector autocorr_detector mlp cnn1d lstm \
  --epochs 3 \
  --batch-size 1024
```

### 4. Train and evaluate CNR-SenseNet

```bash
python -m project.run_cnr_sensenet_eval \
  --epochs 5 \
  --batch-size 1024 \
  --aux-branch-type autocorr \
  --save-checkpoint
```

### 5. Reevaluate a saved checkpoint

```bash
python -m project.evaluate \
  --checkpoint project/plots/cnr_sensenet_eval/cnr_sensenet_checkpoint.pt \
  --dataset-mode bundle
```

## Main Experiment Entrypoints

| Task | Entrypoint | Default output |
| --- | --- | --- |
| Build cached dataset | `python -m project.prepare_dataset` | `project/data/processed/` |
| Train classical / deep baselines | `python -m project.train` | `project/results/baselines/<timestamp>/` |
| Train and evaluate CNR-SenseNet | `python -m project.run_cnr_sensenet_eval` | `project/plots/cnr_sensenet_eval/` |
| Reevaluate a checkpoint | `python -m project.evaluate` | terminal summary or `--output-json` |
| Hyperparameter search | `python -m project.search_cnr_sensenet` | `project/results/cnr_sensenet_search/` |
| Multi-model comparison | `python -m project.run_model_comparison` | `project/plots/model_comparison/` |
| Explainability | `python -m project.explainability` | `project/plots/cnr_sensenet_explainability/` |
| Branch ablation | `python -m project.ablation` | `project/results/cnr_sensenet_ablation/` |
| Robustness evaluation | `python -m project.robustness` | `project/results/cnr_sensenet_robustness/` |

## Registered Models

The current names exposed through `project.models` are:

- `energy_detector`
- `autocorr_detector`
- `mlp`
- `cnn1d`
- `lstm`
- `cnr_sensenet`

This makes it easy to compare classical detectors and neural models under one shared evaluation pipeline.

## Outputs and Artifacts

The repository already includes example outputs such as:

- dataset overview figures: `project/plots/dataset_sample_overview.png`
- main CNR-SenseNet experiment outputs: `project/plots/cnr_sensenet_eval/`
- multi-model comparison outputs: `project/plots/model_comparison/`
- explainability smoke outputs: `project/plots/cnr_sensenet_explainability_smoke/`
- search, ablation, and robustness outputs: `project/results/cnr_sensenet_*/`

Typical outputs include:

- `.json` summaries
- `.csv` metric tables
- `.png` figures
- `.pt` checkpoints

## Cluster Reproduction

The repository already ships a Slurm-ready workflow under:

- `cluster/setup_env.sh`
- `cluster/jobs/*.sbatch`
- `cluster/README.md`

It covers:

- dataset preparation
- baseline training
- main CNR-SenseNet experiments
- hyperparameter search and multi-seed reruns
- model comparison, explainability, ablation, and robustness runs

If your main goal is cluster reproduction, start with `cluster/README.md`.

## Notes

- `project/models/cn_lssnet.py` is kept only as an archived placeholder and is not part of the active experiment pipeline.
- The root `README.md` is the Chinese primary entrypoint. Use the buttons on the first line to switch to the English version.
