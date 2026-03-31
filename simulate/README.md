# simulate

`simulate/` 是为 `CNR-SenseNet` 预留的仿真增强骨架，用来把未来的合成信号、干扰器、信道失真和混合场景整理成一套可复用的数据管线。

当前阶段，这个目录解决两件事：

1. 定义一套与现有 `project/data/*.npz` 兼容的统一 archive schema
2. 提供最小可运行的仿真生成与合并脚本，方便后续逐步扩展

## 目录结构

```text
simulate/
|- core/
|  |- base.py            # 场景配置与批次元数据
|  |- channels.py        # 简单信道变换
|  |- impairments.py     # AWGN、IQ 不平衡、DC 偏置
|  |- interferers.py     # tone / impulsive 干扰器
|  `- composer.py        # QPSK 样本、功率缩放、IQ flatten
|- scripts/
|  |- generate_sim_archive.py   # 生成仿真 archive
|  `- merge_with_rml.py         # 合并真实 archive 与仿真 archive
|- outputs/
|  |- archives/
|  |- manifests/
|  `- previews/
`- schema.py            # 统一 archive 格式与读写工具
```

## 统一 archive 格式

### 必需字段

- `X`: `float32`, shape `[N, 2T]`，交错存储的 IQ 序列，形式为 `[I0, Q0, I1, Q1, ...]`
- `y`: `int64`, shape `[N]`，二分类标签，`1=signal`, `0=noise`
- `snr`: `int64`, shape `[N]`，样本对应的 SNR 标签

### 与现有训练流水线兼容的基础字段

- `mod`: `str`, shape `[N]`
- `label_snr`: `int64`, shape `[N]`
- `source_snr`: `int64`, shape `[N]`
- `sample_type`: `str`, shape `[N]`
- `noise_power`: `float32`, shape `[1]`

### 仿真扩展字段

- `domain`: `str`, shape `[N]`，例如 `real / simulated`
- `scenario`: `str`, shape `[N]`，例如 `qpsk_awgn / qpsk_tone_hardneg`
- `generator`: `str`, shape `[N]`，记录生成脚本或流程名
- `channel_type`: `str`, shape `[N]`
- `interference_type`: `str`, shape `[N]`
- `sample_id`: `str`, shape `[N]`
- `sim_seed`: `int64`, shape `[N]`

设计原则：

- 旧脚本继续读取已有字段，不必立刻修改
- 新脚本可使用扩展字段做溯源、过滤和分层评估
- 仿真数据先独立保存，再与真实数据 archive 合并

## 最小工作流

生成一个仿真 archive：

```bash
python -m simulate.scripts.generate_sim_archive \
  --scenario qpsk_tone_hardneg \
  --num-samples 4096
```

把仿真 archive 合并进真实 archive：

```bash
python -m simulate.scripts.merge_with_rml \
  --base-archive project/data/processed/cluster_job_smoke.npz \
  --sim-archives simulate/outputs/archives/qpsk_tone_hardneg.npz \
  --sim-ratio 0.5 \
  --shuffle
```

## 下一步建议

- 把 `project/robustness.py` 中的测试扰动逐步搬到 `simulate/core/`
- 增加 `configs/`，把不同场景参数从脚本参数中拆出来
- 先做 hard negatives，再做受损正类和多径信道
