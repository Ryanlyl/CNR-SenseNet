# Cluster Runbook

This repository now includes a minimal cluster workflow that matches the NTU EEE GPU cluster rules:

- Do not run heavy jobs on login nodes.
- Use `sbatch` for long-running work.
- When requesting GPUs, always specify a GPU type such as `a5000`, `v100`, `a40`, `l40`, or `6000ada`.
- `module load Miniforge3` and `source activate <env>` are used for Python environments.

The examples below assume:

- cluster login host: `<cluster-login-host>`
- cluster username: `<cluster-user>`
- repo path on cluster: `~/CNR-SenseNet`
- conda env name: `cnr-sensenet`
- cache path on cluster: `~/CNR-SenseNet/project/data/processed/cluster_signal_noise.npz`

## 1. Sync Code From Local To Cluster

Recommended path: push code, then clone or pull on the cluster.

Local PowerShell:

```powershell
cd D:\Dissertation\Code\CNR-SenseNet
git status
git add project cluster
git commit -m "Add cluster workflow"
git push origin <your-branch>
```

Cluster shell:

```bash
ssh <cluster-user>@<cluster-login-host>
git clone https://github.com/Ryanlyl/CNR-SenseNet.git ~/CNR-SenseNet
# or, if the repo already exists:
cd ~/CNR-SenseNet
git pull --ff-only
```

If you do not want to rely on Git, copy the whole repo from local:

```powershell
scp -r D:\Dissertation\Code\CNR-SenseNet <cluster-user>@<cluster-login-host>:~/CNR-SenseNet
```

## 2. Stage Data On The Cluster

The raw `.pkl` is ignored by Git, so data must be copied separately.

Recommended path: copy the prepared `.npz` cache because it is smaller and all cluster scripts can reuse it.

Local PowerShell:

```powershell
scp D:\Dissertation\Code\CNR-SenseNet\project\data\processed\signal_noise_fbe832794a03.npz `
  <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/data/processed/cluster_signal_noise.npz
```

If you need to rebuild the dataset cache on the cluster, also copy the raw pickle:

```powershell
scp D:\Dissertation\Code\CNR-SenseNet\project\data\RML2016.10a_dict.pkl `
  <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/data/RML2016.10a_dict.pkl
```

## 3. Create Or Update The Conda Environment

Cluster shell:

```bash
ssh <cluster-user>@<cluster-login-host>
cd ~/CNR-SenseNet
bash cluster/setup_env.sh
```

## 4. Build The Dataset Cache With Slurm

Use this if you copied the raw `.pkl` and want the cluster to build the reusable cache:

```bash
cd ~/CNR-SenseNet
sbatch --time=00:20:00 cluster/jobs/prepare_dataset.sbatch
```

Example with a forced rebuild:

```bash
sbatch --time=00:20:00 cluster/jobs/prepare_dataset.sbatch --force-rebuild
```

## 5. Train Baselines

This runs `energy_detector`, `autocorr_detector`, `mlp`, `cnn1d`, and `lstm` on the cached `.npz`.

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=08:00:00 cluster/jobs/train_baselines.sbatch --epochs 10 --batch-size 2048
```

Outputs are written under `project/results/baselines/<timestamp>/`.

## 6. Train And Evaluate CNR-SenseNet

This job trains `cnr_sensenet`, exports plots, and saves a checkpoint.

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=08:00:00 cluster/jobs/train_cnr_sensenet.sbatch --epochs 20 --batch-size 2048
```

Outputs are written under `project/plots/cnr_sensenet_eval/`.

## 7. Search CNR-SenseNet Hyperparameters

This job runs the dedicated grid search for `lr`, `weight_decay`, `dropout`, `batch_size`, and `epochs`, and saves the best checkpoint.

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=08:00:00 cluster/jobs/search_cnr_sensenet.sbatch
```

The default search grid matches:

```bash
python -m project.search_cnr_sensenet \
  --device cuda \
  --lr-values 1e-3 5e-4 3e-4 \
  --weight-decay-values 0.0 1e-5 1e-4 \
  --dropout-values 0.1 0.2 0.3 \
  --batch-size-values 512 1024 2048 \
  --epochs-values 10 20 30 \
  --patience 3 \
  --save-best-checkpoint \
  --output-dir project/results/cnr_sensenet_search_gpu \
  --output-prefix gpu_search
```

Outputs are written under `project/results/cnr_sensenet_search_gpu/`.
If the job hits the wall time, resubmitting the same command with the same output directory will resume from the finished trials already listed in `gpu_search_results.csv`.

## 8. Run The Round-2 Narrow Search

This job narrows the search around the current best region found on the first pass:

- `lr`: `8e-4 1e-3 1.2e-3`
- `weight_decay`: `0.0 1e-6 1e-5`
- `dropout`: `0.15 0.2 0.25`
- `batch_size`: `256 512 768`
- `epochs`: `30 40`

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=10:00:00 cluster/jobs/search_cnr_sensenet_narrow.sbatch
```

Outputs are written under `project/results/cnr_sensenet_search_round2/`.
Like the first-pass search, resubmitting the same command with the same output directory will resume from the finished trials already listed in `gpu_search_round2_results.csv`.

## 9. Re-Run The Current Best Config Across Multiple Seeds

This job repeats the current best setting on several seeds to check stability:

- default seeds: `42 43 44`
- default config: `lr=1e-3`, `weight_decay=0.0`, `dropout=0.2`, `batch_size=512`, `epochs=30`, `patience=3`

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=06:00:00 cluster/jobs/train_cnr_sensenet_multiseed.sbatch
```

Outputs are written under `project/results/cnr_sensenet_multiseed/seed_<seed>/`.

You can override the seeds or defaults with environment variables, for example:

```bash
SEEDS="52 62 72" OUTPUT_DIR=~/CNR-SenseNet/project/results/cnr_sensenet_multiseed_alt \
sbatch --gpus=a5000:1 --time=06:00:00 cluster/jobs/train_cnr_sensenet_multiseed.sbatch
```

## 10. Run Robustness Evaluation

This job trains the clean `cnr_sensenet` model once, then evaluates the saved threshold under the configured test-time perturbations.

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=08:00:00 cluster/jobs/robustness_cnr_sensenet.sbatch
```

Outputs are written under `project/results/cnr_sensenet_robustness/`.

Recommended paper-facing file names inside that directory are:

- `cnr_sensenet_robustness_table_robustness_1_overall.csv`
- `cnr_sensenet_robustness_table_robustness_2_by_snr.csv`
- `cnr_sensenet_robustness_figure_robustness_1_main.png`
- `cnr_sensenet_robustness_figure_robustness_2_worst_case_snr.png`

You can override the perturbation grid or output directory with environment variables, for example:

```bash
OUTPUT_DIR=~/CNR-SenseNet/project/results/cnr_sensenet_robustness_impulse_sweep \
AWGN_STD_VALUES="0.05 0.1 0.2 0.4" \
IMPULSE_PROB_VALUES="0.01 0.05 0.1 0.15" \
sbatch --gpus=a5000:1 --time=08:00:00 cluster/jobs/robustness_cnr_sensenet.sbatch
```

## 11. Run Multi-Model Comparison

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=12:00:00 cluster/jobs/model_comparison.sbatch --epochs 10 --batch-size 2048
```

Outputs are written under `project/plots/model_comparison/`.

## 12. Evaluate A Saved Checkpoint

`project/evaluate.py` now supports checkpoint evaluation for saved `.pt` files.

Baseline checkpoint example:

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=01:00:00 cluster/jobs/evaluate_checkpoint.sbatch \
  ~/CNR-SenseNet/project/results/baselines/20260323_194038/mlp.pt
```

CNR-SenseNet checkpoint example:

```bash
cd ~/CNR-SenseNet
sbatch --gpus=a5000:1 --time=01:00:00 cluster/jobs/evaluate_checkpoint.sbatch \
  ~/CNR-SenseNet/project/plots/cnr_sensenet_eval/cnr_sensenet_checkpoint.pt \
  --dataset-mode bundle
```

You can override defaults the same way as the original scripts, for example:

```bash
sbatch --gpus=v100:1 --time=01:00:00 cluster/jobs/evaluate_checkpoint.sbatch \
  ~/CNR-SenseNet/project/results/baselines/20260323_194038/cnn1d.pt \
  --threshold 0.55 --test-ratio 0.2 --val-ratio 0.1 --seed 42
```

## 13. Watch Jobs And Inspect Logs

```bash
squeue -u $USER
tail -f slurm-cnr-baselines-<jobid>.out
tail -f slurm-cnr-train-<jobid>.out
tail -f slurm-cnr-search-<jobid>.out
tail -f slurm-cnr-search-r2-<jobid>.out
tail -f slurm-cnr-multiseed-<jobid>.out
tail -f slurm-cnr-robust-<jobid>.out
```

The cluster docs also recommend using Slurm commands such as:

```bash
sacct -j <jobid>
```

## 14. Copy Results Back To Local

Local PowerShell:

```powershell
scp -r <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/results/baselines `
  D:\Dissertation\Code\CNR-SenseNet\project\results\cluster-baselines
```

```powershell
scp -r <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/plots/cnr_sensenet_eval `
  D:\Dissertation\Code\CNR-SenseNet\project\plots\cnr_sensenet_eval_cluster
```

```powershell
scp -r <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/results/cnr_sensenet_robustness `
  D:\Dissertation\Code\CNR-SenseNet\project\results\cnr_sensenet_robustness_cluster
```

```powershell
scp -r <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/results/cnr_sensenet_search_gpu `
  D:\Dissertation\Code\CNR-SenseNet\project\results\cnr_sensenet_search_gpu_cluster
```

```powershell
scp -r <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/results/cnr_sensenet_search_round2 `
  D:\Dissertation\Code\CNR-SenseNet\project\results\cnr_sensenet_search_round2_cluster
```

```powershell
scp -r <cluster-user>@<cluster-login-host>:~/CNR-SenseNet/project/results/cnr_sensenet_multiseed `
  D:\Dissertation\Code\CNR-SenseNet\project\results\cnr_sensenet_multiseed_cluster
```

## Notes

- Your current raw pickle is about 641 MB and the existing processed caches are about 426 MB total, so keeping this project under `/home` is reasonable for now.
- If later runs accumulate many checkpoints and plots, move `project/results` or `project/plots` into `/projects/...` and point `OUTPUT_DIR` there when submitting jobs.
- For preemptible runs you can add `--qos=override-limits-but-killable` to `sbatch`, but only if your training code can tolerate interruption.
