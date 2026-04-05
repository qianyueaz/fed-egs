# Fed-EGS

PyTorch implementation of a Federated Expert-General Subnetwork validation framework on CIFAR-10.

## What is included

- WidthScalableResNet18 with width 1.0 general model and width 0.25 expert subnet
- Static supernet-to-subnet weight slicing and server-side expert delta aggregation
- CIFAR-10 train/test difficulty scoring, persistent client partition caching, and a reusable public distillation split
- Independent algorithm files for `FedAvg`, `FedProx`, `Fed-EGS`, `FedEGS-2`, and `FedEGS-3`
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

- ```
  fed-egs/
  ├── configs/
  │   ├── fedavg_match_fedegs2_expert_cifar10.yaml
  │   ├── fedavg_match_fedegs_expert_cifar10.yaml
  │   ├── fedegs2_cifar10.yaml
  │   ├── fedegs3_cifar10.yaml new53L
  │   ├── fedegs_cifar10.yaml
  │   ├── fedegs_v1_cifar10.yaml
  │   ├── fedprox_match_fedegs2_expert_cifar10.yaml
  │   └── fedprox_match_fedegs_expert_cifar10.yaml
  │
  ├── fedegs/
  │   ├── __init__.py 2L
  │   ├── config.py modified156L+distill_alpha, +prototype_weight
  │   ├── evaluation.py 38L
  │   ├── experiment.py 86L
  │   │
  │   ├── data/
  │   │   ├── __init__.py 3L
  │   │   └── cifar10_federated.py 397L数据划分/难度评分/公共集
  │   │
  │   ├── federated/
  │   │   ├── __init__.py 4L
  │   │   ├── common.py 391LBaseFederatedClient/Server, 双阈值路由
  │   │   ├── factory.py modified20L+fedegs3 注册
  │   │   ├── client.py 3L
  │   │   ├── server.py 4L
  │   │   │
  │   │   └── algorithms/
  │   │       ├── __init__.py 5L
  │   │       ├── fedavg.py 92LFedAvg 基线
  │   │       ├── fedprox.py 95LFedProx 基线
  │   │       ├── fedegs.py 206Lv1: 超网切片 + delta 聚合
  │   │       ├── fedegs2.py 218Lv2: SmallCNN + 公共集 logits 蒸馏
  │   │       └── fedegs3.py new501Lv3: 双向蒸馏 + 类别原型正则
  │   │
  │   └── models/
  │       ├── __init__.py 53Lbuild_model / build_baseline_model
  │       ├── small_cnn.py 33L3层轻量CNN (FedEGS-2/3 专家)
  │       └── width_scalable_resnet.py 215L可缩放宽度ResNet18 (通用模型/超网)
  │
  ├── train.py 155L主入口: 配置加载 → 数据 → 实验 → 日志
  ├── pretrain.py 123L难度评分预训练
  ├── requirements.txt 5L
  ├── README.md modified95L
  └── FED_EGS_ARCHITECTURE.md modified87L
  ```

- Algorithm implementations:
  
  - `fedegs/federated/algorithms/fedegs.py`
  - `fedegs/federated/algorithms/fedavg.py`
  - `fedegs/federated/algorithms/fedprox.py`
  - `fedegs/federated/algorithms/fedegs2.py`
  - `fedegs/federated/algorithms/fedegs3.py`
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

## FedEGS-3

`FedEGS-3` extends FedEGS-2 with **mutual knowledge distillation** and **class-wise logit prototypes**:

1. **Local knowledge distillation**: The server's general model is sent to clients and used as a teacher during expert training. The expert loss becomes `(1-α)·CE + α·KL(expert, teacher)`, where only a forward pass through the teacher is needed (no backprop on the large model).
2. **Class prototypes**: Each client computes per-class average logits over its local data and uploads them. The server aggregates prototypes and adds an MSE regularisation term during general model distillation, aligning the general model's outputs with the true local class distributions.

This creates a bidirectional knowledge loop: the general model improves the experts via soft labels, and the experts improve the general model via public logits and prototypes.

Example:

```bash
python train.py --config configs/fedegs3_cifar10.yaml
```

Key hyperparameters:

```yaml
federated:
  server_algorithm: fedegs3
  distill_alpha: 0.5        # weight of KD loss in local expert training
  prototype_weight: 0.1     # weight of prototype regularisation on server
  distill_temperature: 2.0  # temperature for both local and server KD
```
