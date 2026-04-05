# Fed-EGS Architecture

## Separation of Responsibilities

The project now separates two concerns clearly:

### 1. Algorithm implementation

Each algorithm file only contains its own client/server logic and self-evaluation:
- `fedegs/federated/algorithms/fedegs.py`
- `fedegs/federated/algorithms/fedavg.py`
- `fedegs/federated/algorithms/fedprox.py`
- `fedegs/federated/algorithms/fedegs2.py`

### 2. Experiment orchestration

Cross-algorithm comparison is handled by:
- `fedegs/experiment.py`

That runner uses one shared config and one shared data split, then executes multiple algorithms in sequence and aggregates results.

## Current Flow

1. `train.py` loads the YAML config.
2. Data is prepared once.
3. `run_experiment_suite(...)` runs the primary algorithm.
4. The same runner executes comparison algorithms listed in config.
5. Results are merged into one experiment result object.

## Why This Is Better

- `fedegs.py` no longer knows anything about `FedAvg` or `FedProx`.
- Adding a new algorithm only requires:
  1. a new algorithm file
  2. a factory registration
  3. optionally adding it to `compare_algorithms`
- Comparison logic is centralized in one place instead of leaking into every algorithm.

## Execution Example

```powershell
activate pytorch
python train.py --config configs/fedegs_cifar10.yaml
```

With config:

```yaml
federated:
  server_algorithm: fedegs
  compare_algorithms:
    - fedavg
    - fedprox
```

This means:
- `Fed-EGS` is the primary run
- `FedAvg` and `FedProx` are comparison runs
- all three share the same dataset preparation and experiment hyperparameters


## FedEGS-2 Flow

`FedEGS-2` targets the convergence issue of blockwise supernet updates. Its structure is:

1. Each client trains a small personalized CNN expert on its private local data.
2. A small public CIFAR-10 split is shared by all participants.
3. Sampled clients run inference on the public split and upload logits instead of model deltas.
4. The server averages those logits and distills the large general model on the public split.
5. Inference still uses the original confidence routing: local expert first, then fallback to the general model for low-confidence samples.

This keeps client-side training lightweight while giving the general model a denser, more stable supervision signal than sparse block updates.


## FedEGS-3 Flow

`FedEGS-3` adds mutual knowledge distillation on top of FedEGS-2:

1. The server sends the current general model to sampled clients.
2. Each client uses the general model as a **teacher** (forward-only, no backprop) to produce soft labels on its private local data.
3. The client trains its small expert with a combined loss: `(1-α)·CE(expert, hard_label) + α·KL(expert, teacher_soft_label)`.
4. The client computes **class-wise logit prototypes** (per-class average logits over local data) and uploads them alongside public-set logits.
5. The server aggregates public logits (weighted average) and class prototypes (weighted average).
6. The server distills the general model on the public split with an additional **prototype regularisation** term: `MSE(general_class_mean_logit, aggregated_prototype)`.
7. Inference uses the same dual-threshold routing as FedEGS-2.

This creates a bidirectional knowledge loop: general → expert (via soft labels) and expert → general (via logits + prototypes).
