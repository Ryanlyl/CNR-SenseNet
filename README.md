# CNR-SenseNet

当前仓库以 `RML2016.10a` 为基础，围绕二分类 `signal vs noise` 检测任务搭建了一套统一的数据、模型和实验框架。现阶段重点是：

- 用 `RML2016.10a_dict.pkl` 重建带缓存的数据集
- 同时保留后续扩展到 `AMC + Spectrum Sensing` 的标签设计
- 统一 `models/` 接口，方便后续接入 `ED / MLP / CNN / LSTM / Ours`
- 提供训练、可视化和 CNR-SenseNet 专用评估脚本

## 当前目录

```text
project/
  data/
    RML2016.10a_dict.pkl
    gen_dataset.py
    visualize_dataset_samples.py
    dataset_represent_201610a.py
    processed/
  models/
    __init__.py
    base.py
    torch_binary.py
    ed.py
    mlp.py
    cnn1d.py
    lstm.py
    cn_lssnet.py
  CNR_SenseNet.py
  train.py
  run_cnr_sensenet_eval.py
  evaluate.py
  ablation.py
  robustness.py
  utils.py
  plots/
  results/
```

## 数据模块怎么用

### 1. 原始数据说明

原始数据文件在：

- `project/data/RML2016.10a_dict.pkl`

这个文件来自 `RML2016.10a`，原始键格式是 `(mod, snr)`，值是对应调制和 SNR 条件下的 IQ 样本。

### 2. 重建后的标签设计

重建后的融合数据集包含以下标签：

- `y`
  - 二分类标签
  - `1 = signal`
  - `0 = noise`
- `mod`
  - 正样本保留原始调制标签
  - 负样本统一标为 `noise`
- `label_snr`
  - 正样本保留原始 SNR 标签
  - 负样本统一标为 `0`
- `source_snr`
  - 表示这个样本对应的原始信号 SNR
  - 正样本和负样本都会保留
  - 用于按 SNR 分层和后续 `Pd vs SNR` 评估

注意：

- 当前 `SignalNoiseDataset.__getitem__()` 返回的是 `(x, y, snr)`，其中这个 `snr` 实际上是 `source_snr`
- 如果要读完整标签，可以用 `dataset.get_labels(idx)` 或 `DatasetBundle.train_arrays / test_arrays`

### 3. `project/data/gen_dataset.py`

这是当前最重要的数据脚本，负责：

- 将 `[2, 128]` IQ 样本交错成 `[256]`
- 假设归一化样本为“单位功率附近”的数据
- 固定噪声功率 `noise_power=1.0`
- 根据样本的 `SNR` 将正样本恢复到目标总功率
- 为每个正样本生成对应的固定功率 AWGN 负样本
- 按 `binary label + source_snr` 分层切分
- 将融合后的数据缓存到 `project/data/processed/*.npz`

主要函数：

- `build_signal_vs_noise_dataset(...)`
  - 从原始 `pkl` 生成融合数据
- `prepare_signal_vs_noise_dataset(...)`
  - 优先加载缓存；没有缓存时再构建并保存
- `load_signal_vs_noise_archive(...)`
  - 直接加载 `.npz` 缓存
- `stratified_split_binary(...)`
  - 按 `label + source_snr` 分层

直接生成或刷新缓存：

```bash
python -m project.data.gen_dataset
```

如果你在代码里调用，推荐这样用：

```python
from project.data import DataConfig, build_datasets

bundle = build_datasets(
    DataConfig(
        noise_power=1.0,
        use_cache=True,
        force_rebuild=False,
    )
)
```

### 4. `project/data/__init__.py`

这里提供标准数据入口，建议其他模块都从这里进，不要直接各写各的加载逻辑。

主要对象：

- `DataConfig`
  - 控制 `pkl_path`
  - `test_ratio`
  - `seed`
  - `noise_power`
  - `snr_filter`
  - `selected_mods`
  - `cache_path`
  - `use_cache`
  - `force_rebuild`
- `build_datasets(config)`
  - 返回 `DatasetBundle`
- `DatasetBundle`
  - `train_dataset`
  - `test_dataset`
  - `train_arrays`
  - `test_arrays`
  - `train_meta`
  - `test_meta`
  - `mods`
  - `snrs`
  - `cache_path`

### 5. `project/data/visualize_dataset_samples.py`

这个脚本用于从重建后的数据集中自动挑选代表性样本并画图。

当前默认会：

- 同时挑 `signal` 和 `noise`
- 同时覆盖低 / 中 / 高 SNR
- 尽量覆盖几种典型调制
- 生成 10 个样本的总览图
- 每个子图标注 `type / mod / label_snr / source_snr`

导出 PNG：

```bash
python -m project.data.visualize_dataset_samples
```

导出 PDF：

```bash
python -m project.data.visualize_dataset_samples --output project/plots/dataset_sample_overview.pdf
```

### 6. `project/data/dataset_represent_201610a.py`

这个脚本只是用来快速查看原始 `RML2016.10a` 的结构，不参与当前训练流程。

用途：

- 查看有哪些 modulation
- 查看有哪些 SNR
- 查看原始样本形状

## 模型模块怎么用

### 1. `project/models/base.py`

这是所有模型统一遵守的基类接口：

- `fit(train_dataset, val_dataset=None, **kwargs)`
- `predict_scores(dataset)`
- `predict(dataset, threshold=...)`
- `get_config()`

后面你继续加模型时，最好都继承这个接口，避免训练和评估脚本分裂。

### 2. `project/models/__init__.py`

这是模型注册表和工厂入口。

核心用法：

```python
from project.models import create_model

model = create_model("mlp", epochs=5, batch_size=1024)
```

当前注册的名字包括：

- `ed`
- `mlp`
- `cnn1d`
- `lstm`
- `cnr_sensenet`
- `cn_lssnet`

### 3. `project/models/ed.py`

这是 `SNRAdaptiveEnergyDetector`，主要用于做传统 `ED` baseline。

当前特点：

- 按 `source_snr` 为每个 SNR 拟合一个阈值
- 支持 `sum` 或 `mean` 统计量
- 支持 `balanced_acc / youden / target_pfa`
- 可以按 SNR 评估 `Pd / Pfa / Acc`

典型用法：

```python
from project.data import DataConfig, build_datasets
from project.models import create_model

bundle = build_datasets(DataConfig(noise_power=1.0))
model = create_model("ed", noise_power=1.0, statistic="sum", thr_mode="balanced_acc")
model.fit(bundle.train_dataset)
results = model.evaluate_by_snr(bundle.test_dataset)
```

### 4. `project/models/torch_binary.py`

这是 `MLP / CNN1D / LSTM / CNR-SenseNet` 这类 PyTorch 二分类模型共用的训练器。

它负责：

- DataLoader 构建
- BCE loss
- 训练循环
- 验证集 early stopping
- `predict_scores()`
- `state_dict()` / `load_state_dict()`

如果后面还有新的深度学习 baseline，优先复用这里。

### 5. `project/models/mlp.py`

MLP 基线，输入是 `[256]` 的交错 IQ 向量。

适合：

- 做最直接的全连接基线
- 快速验证数据和训练流程是否正常

### 6. `project/models/cnn1d.py`

1D CNN 基线，直接对交错后的序列做卷积。

适合：

- 做时域局部模式提取
- 作为比 MLP 更强的简单卷积基线

### 7. `project/models/lstm.py`

LSTM 基线，会把 `[256]` 重塑成 `128 x 2` 的 IQ 序列再建模。

适合：

- 利用时序依赖
- 作为 RNN 类对照基线

### 8. `project/CNR_SenseNet.py`

这是当前 `Ours / CNR-SenseNet` 的主实现文件，已经注册为 `cnr_sensenet`。

如果你想单独训练或评估这个模型，优先使用下面的专用脚本：

- `project/run_cnr_sensenet_eval.py`

### 9. `project/models/cn_lssnet.py`

目前还是占位文件，接口已经保留，但还没有实现。

## 训练和实验入口怎么用

### 1. `project/train.py`

这是当前通用训练入口，主要面向 PyTorch 类模型。

推荐当前使用：

- `mlp`
- `cnn1d`
- `lstm`
- `cnr_sensenet`

示例：

```bash
python -m project.train --models mlp cnn1d lstm --epochs 3 --batch-size 1024 --output-dir project/results/baselines
```

如果你想指定缓存数据集：

```bash
python -m project.train --dataset-npz project/data/processed/signal_noise_fbe832794a03.npz --models mlp cnn1d lstm
```

说明：

- 这个脚本默认从 `project/data/processed/` 里找 `.npz`
- 会自动做 `train / val / test` 划分
- 会输出权重、JSON 指标和汇总表

### 2. `project/run_cnr_sensenet_eval.py`

这是 `CNR-SenseNet` 的专用训练评估脚本，功能最完整。

它会：

- 用统一数据入口加载数据
- 训练 `cnr_sensenet`
- 评估整体指标
- 输出按 SNR 的指标
- 画 ROC / PR / confusion matrix / training history / SNR 曲线

示例：

```bash
python -m project.run_cnr_sensenet_eval --epochs 5 --batch-size 1024 --save-checkpoint
```

### 3. `project/evaluate.py`

目前还是占位入口，后续计划用于：

- 加载已有 checkpoint
- 复现主结果图和主结果表

### 4. `project/ablation.py`

目前还是占位入口，后续计划用于：

- 跑模块消融
- 导出 ablation table

### 5. `project/robustness.py`

目前还是占位入口，后续计划用于：

- 跑鲁棒性实验
- 导出 robustness table / plots

### 6. `project/utils.py`

当前主要提供路径辅助函数：

- `resolve_path(...)`

如果后面要加统一日志、随机种子、结果导出等通用工具，也建议放这里。

## 结果文件在哪里

### 1. 数据缓存

- `project/data/processed/*.npz`

### 2. 可视化图

- `project/plots/`

例如当前已经有：

- `project/plots/dataset_sample_overview.png`
- `project/plots/dataset_sample_overview.pdf`

### 3. 训练和评估输出

- `project/results/`

例如：

- `project/results/baselines/`
- `project/plots/cnr_sensenet_eval/`

## 当前建议的使用顺序

如果你是第一次进这个仓库，建议按下面顺序看：

1. `project/data/gen_dataset.py`
2. `project/data/__init__.py`
3. `project/models/base.py`
4. `project/models/__init__.py`
5. `project/models/ed.py`
6. `project/models/torch_binary.py`
7. `project/models/mlp.py / cnn1d.py / lstm.py`
8. `project/CNR_SenseNet.py`
9. `project/train.py`
10. `project/run_cnr_sensenet_eval.py`

## 当前状态总结

已经可用：

- 原始 `RML2016.10a` 到融合数据集的构建与缓存
- 带 `mod / label_snr / source_snr` 的标签设计
- 代表性样本可视化导出为 PNG / PDF
- `ED` 传统基线
- `MLP / CNN1D / LSTM / CNR-SenseNet` 深度学习训练框架

仍待继续：

- `CN-LSSNet` 实现
- 通用 `evaluate.py`
- `ablation.py`
- `robustness.py`
- 主结果表、主结果图的统一自动导出
