[![README Chinese](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-0F766E?style=for-the-badge)](./README.md)
[![README English](https://img.shields.io/badge/README-English-1D4ED8?style=for-the-badge)](./README.en.md)

# CNR-SenseNet

`CNR-SenseNet` 是一个围绕 `RML2016.10a` 构建的二分类频谱检测研究仓库，目标是统一完成 `signal vs noise` 数据准备、传统基线、深度学习模型、论文型实验以及 GPU Cluster 复现流程。

## 项目亮点

- 统一的数据入口：从原始 `RML2016.10a_dict.pkl` 重建并缓存 `signal vs noise` 数据集，支持按 `SNR` 和调制方式筛选。
- 统一的检测器接口：传统检测器和深度学习模型都遵循同一套 `fit / predict_scores / predict` 流程。
- 完整的研究链路：仓库内已经覆盖训练、复评、超参搜索、模型对比、可解释性、消融、鲁棒性与 Slurm 工作流。
- 可复现实验产物：`project/plots/` 和 `project/results/` 中已经包含多组示例图表、JSON 摘要和 checkpoint。

## CNR-SenseNet 结构

`project/CNR_SenseNet.py` 中的主模型由三类分支组成，并支持在消融实验中按分支开关：

- `Raw Branch`：直接从原始 IQ 序列提取时域特征。
- `Energy Branch`：从局部能量窗口提取统计特征。
- `Aux Branch`：可切换为 `Diff` 或 `Autocorr`，用于建模相邻样本差分或自相关线索。

## 仓库结构

```text
CNR-SenseNet/
|- project/
|  |- data/                     # 原始数据、缓存数据、数据构建与可视化
|  |- models/                   # Energy / Autocorr / MLP / CNN1D / LSTM 等模型
|  |- CNR_SenseNet.py           # 主模型实现
|  |- prepare_dataset.py        # 生成或复用缓存数据集
|  |- train.py                  # 训练基线模型
|  |- run_cnr_sensenet_eval.py  # 训练并评估 CNR-SenseNet
|  |- evaluate.py               # 对已有 checkpoint 做复评
|  |- search_cnr_sensenet.py    # 网格搜索
|  |- run_model_comparison.py   # 多模型统一对比
|  |- explainability.py         # 可解释性分析
|  |- ablation.py               # 分支消融实验
|  |- robustness.py             # 鲁棒性评估
|  |- plots/                    # 图表输出
|  `- results/                  # 搜索、消融、鲁棒性等结果
`- cluster/
   |- environment.yml           # Conda 环境定义
   |- setup_env.sh              # Cluster 环境初始化脚本
   |- jobs/*.sbatch             # Slurm 作业模板
   `- README.md                 # Cluster 使用说明
```

## 快速开始

### 1. 准备环境

推荐直接复用 `cluster/environment.yml`：

```bash
conda env create -f cluster/environment.yml
conda activate cnr-sensenet
```

说明：

- 该环境文件默认包含 `pytorch-cuda=12.1`，更适合 GPU / Cluster 场景。
- 如果你是纯 CPU 本地环境，可以按同样依赖手动调整 PyTorch 安装方式。

### 2. 准备数据

将原始数据放到：

```text
project/data/RML2016.10a_dict.pkl
```

然后生成或复用缓存：

```bash
python -m project.prepare_dataset
```

如果你暂时没有原始 `.pkl`，也可以直接使用仓库现有的缓存文件，并在训练脚本里通过 `--dataset-npz` 指定 `project/data/processed/*.npz`。

### 3. 训练基线模型

```bash
python -m project.train \
  --models energy_detector autocorr_detector mlp cnn1d lstm \
  --epochs 3 \
  --batch-size 1024
```

### 4. 训练并评估 CNR-SenseNet

```bash
python -m project.run_cnr_sensenet_eval \
  --epochs 5 \
  --batch-size 1024 \
  --aux-branch-type autocorr \
  --save-checkpoint
```

### 5. 复评已有 checkpoint

```bash
python -m project.evaluate \
  --checkpoint project/plots/cnr_sensenet_eval/cnr_sensenet_checkpoint.pt \
  --dataset-mode bundle
```

## 常用实验入口

| 任务 | 入口脚本 | 默认输出位置 |
| --- | --- | --- |
| 构建缓存数据集 | `python -m project.prepare_dataset` | `project/data/processed/` |
| 训练传统/深度基线 | `python -m project.train` | `project/results/baselines/<timestamp>/` |
| 训练并评估 CNR-SenseNet | `python -m project.run_cnr_sensenet_eval` | `project/plots/cnr_sensenet_eval/` |
| 复评 checkpoint | `python -m project.evaluate` | 终端摘要或 `--output-json` 指定位置 |
| 超参数搜索 | `python -m project.search_cnr_sensenet` | `project/results/cnr_sensenet_search/` |
| 多模型对比 | `python -m project.run_model_comparison` | `project/plots/model_comparison/` |
| 可解释性分析 | `python -m project.explainability` | `project/plots/cnr_sensenet_explainability/` |
| 分支消融 | `python -m project.ablation` | `project/results/cnr_sensenet_ablation/` |
| 鲁棒性评估 | `python -m project.robustness` | `project/results/cnr_sensenet_robustness/` |

## 当前支持的模型

通过 `project.models` 注册的模型名包括：

- `energy_detector`
- `autocorr_detector`
- `mlp`
- `cnn1d`
- `lstm`
- `cnr_sensenet`

这意味着你可以在统一脚本中对经典检测器和神经网络做同口径比较。

## 结果与产物

仓库里已经包含可直接查看的示例产物，例如：

- 数据样本概览：`project/plots/dataset_sample_overview.png`
- CNR-SenseNet 主实验：`project/plots/cnr_sensenet_eval/`
- 多模型对比：`project/plots/model_comparison/`
- 可解释性示例：`project/plots/cnr_sensenet_explainability_smoke/`
- 消融、搜索、鲁棒性结果：`project/results/cnr_sensenet_*/`

这些目录通常会同时输出：

- `.json` 摘要
- `.csv` 指标表
- `.png` 图表
- `.pt` checkpoint

## Cluster 复现

仓库已提供完整的 Slurm 工作流，入口位于：

- `cluster/setup_env.sh`
- `cluster/jobs/*.sbatch`
- `cluster/README.md`

适合的任务包括：

- 数据缓存构建
- baseline 训练
- CNR-SenseNet 主实验
- 超参搜索与多 seed 复现
- 模型对比、可解释性、消融与鲁棒性评估

如需直接按 GPU Cluster 复现，优先阅读 `cluster/README.md`。

## 适合谁使用

- 想快速复现 `signal vs noise` 检测实验的研究者
- 想比较传统检测器与深度学习模型的同学
- 需要将本地实验平滑迁移到 Slurm / GPU Cluster 的论文项目

## 说明

- `project/models/cn_lssnet.py` 目前仅作为历史占位文件保留，不属于当前活跃实验管线。
- 根目录 `README.md` 为中文版主入口；如需英文版，可使用文档顶部按钮一键切换。
