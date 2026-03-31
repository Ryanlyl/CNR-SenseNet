# deploy

`deploy/` stores the edge-deployment track for `CNR-SenseNet`, separate from the research and training pipeline under `project/`.

## Scope

This directory is intentionally lightweight. The first goal is to keep edge deployment work isolated while we stabilize:

- model export
- runtime loading and inference
- CPU latency benchmarking
- future Raspberry Pi / SDR integration

## Files

- `export_torchscript.py`: exports a training checkpoint to `TorchScript` plus `metadata.json`
- `runtime.py`: shared runtime helpers for metadata loading, input normalization, and batch inference
- `infer.py`: CLI entrypoint for offline inference on `.npy` / `.npz`
- `benchmark.py`: simple CPU latency benchmark for exported artifacts

## Expected input layout

The current runtime assumes the same signal layout used in training:

- flattened interleaved IQ: `[I0, Q0, I1, Q1, ...]`
- default length: `256`
- equivalent IQ shape: `[2, 128]`

`infer.py` and `runtime.py` accept these shapes:

- `[256]`
- `[batch, 256]`
- `[2, 128]`
- `[batch, 2, 128]`

## Typical workflow

Export a checkpoint:

```bash
python -m deploy.export_torchscript \
  --checkpoint project/results/cnr_sensenet_eval_smoke/smoke_eval_checkpoint.pt \
  --output-dir deploy/artifacts/cnr_sensenet_cpu
```

Run offline inference:

```bash
python -m deploy.infer \
  --model deploy/artifacts/cnr_sensenet_cpu/model.torchscript.pt \
  --input project/data/processed/signal_noise_21473ef8892d.npz
```

Benchmark CPU latency:

```bash
python -m deploy.benchmark \
  --model deploy/artifacts/cnr_sensenet_cpu/model.torchscript.pt \
  --batch-size 1 \
  --iterations 200
```

## Next steps

Suggested follow-up work for this directory:

1. add a dedicated edge artifact packer for checkpoint + metadata + version info
2. add a streaming inference loop for SDR or recorded IQ chunks
3. add post-training quantization and Raspberry Pi validation scripts
