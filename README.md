# Fed-EGS

PyTorch implementation of a Federated Expert-General Subnetwork validation framework on CIFAR-10.

## What is included

- WidthScalableResNet18 with width 1.0 general model and width 0.25 expert subnet
- Static supernet-to-subnet weight slicing and server-side expert delta aggregation
- CIFAR-10 train/test difficulty scoring, persistent client partition caching, and a reusable public distillation split
- Independent algorithm files for `FedAvg`, `FedProx`, `Fed-EGS`, and `FedEGS-2`
- A unified experiment runner that executes multiple algorithms with one shared config
- Time-stamped local file logging, clearly named TensorBoard runs, and JSON experiment export
- Single-process sequential client simulation for low-memory devices
- YAML config-driven experiments with optional CLI overrides

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py --config configs/fedegs_cifar10.yaml
```

## Structure

- Algorithm implementations:
  - `fedegs/federated/algorithms/fedegs.py`
  - `fedegs/federated/algorithms/fedavg.py`
  - `fedegs/federated/algorithms/fedprox.py`
  - `fedegs/federated/algorithms/fedegs2.py`
- Unified experiment runner:
  - `fedegs/experiment.py`

## Outputs

- Log file: `artifacts/logs/YYYYMMDD/<experiment>_<algorithm>_YYYYMMDD_HHMMSS.log`
- TensorBoard: `tensorboard --logdir artifacts/tensorboard` and each run is stored under `artifacts/tensorboard/<experiment>/<experiment>_<algorithm>_YYYYMMDD_HHMMSS/`
- Cached difficulty splits and client partitions: `artifacts/cache/`
- Experiment results: `artifacts/experiment_results.json`

## Notes

- Each algorithm file now evaluates only itself.
- Cross-algorithm comparison is handled outside the algorithms by the unified experiment runner.
- The default setup keeps `num_workers=0` so data loading stays single-process and works on restricted Windows environments.

## FedEGS-2

`FedEGS-2` keeps the original routed inference policy, but replaces sliced expert training with a compact personalized CNN on each client. Clients upload logits on a small public CIFAR-10 subset, and the server distills the general `WidthScalableResNet18` on that public data.

Example:

```bash
python train.py --config configs/fedegs_cifar10.yaml --experiment-name fedegs2_trial
```

Use this config change:

```yaml
federated:
  server_algorithm: fedegs2

dataset:
  public_dataset_size: 1000
```
