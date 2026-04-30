import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from fedegs.federated.common import RoundMetrics


def save_metrics(output_dir: str, history: List[RoundMetrics], results: Dict[str, object]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    serializable_history = [{"round": item.round_idx, **asdict(item)} for item in history]
    (path / "round_metrics.json").write_text(json.dumps(serializable_history, indent=2), encoding="utf-8")
    (path / "experiment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def format_memory_table(memory_metrics: Dict[str, float]) -> str:
    if not memory_metrics or "expert" not in memory_metrics or "general" not in memory_metrics:
        return "| Model | Train Memory (MB) | Inference Memory (MB) |\n| --- | ---: | ---: |\n| N/A | N/A | N/A |\n"

    expert = memory_metrics["expert"]
    general = memory_metrics["general"]
    if isinstance(expert, dict) and isinstance(general, dict):
        expert_train = float(expert.get("train", 0.0))
        expert_infer = float(expert.get("infer", 0.0))
        expert_train_peak = float(expert.get("train_peak", 0.0))
        expert_infer_peak = float(expert.get("infer_peak", 0.0))
        general_train = float(general.get("train", 0.0))
        general_infer = float(general.get("infer", 0.0))
        general_train_peak = float(general.get("train_peak", 0.0))
        general_infer_peak = float(general.get("infer_peak", 0.0))
        train_ratio = general_train / max(expert_train, 1e-8)
        infer_ratio = general_infer / max(expert_infer, 1e-8)
        return (
            "| Model | Theory Train (MB) | Theory Infer (MB) | Peak Train (MB) | Peak Infer (MB) |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
            f"| Expert | {expert_train:.2f} | {expert_infer:.2f} | {expert_train_peak:.2f} | {expert_infer_peak:.2f} |\n"
            f"| General | {general_train:.2f} | {general_infer:.2f} | {general_train_peak:.2f} | {general_infer_peak:.2f} |\n"
            f"| Ratio (General / Expert) | {train_ratio:.2f}x | {infer_ratio:.2f}x | {general_train_peak / max(expert_train_peak, 1e-8):.2f}x | {general_infer_peak / max(expert_infer_peak, 1e-8):.2f}x |\n"
        )

    expert_value = float(expert)
    general_value = float(general)
    ratio = general_value / max(expert_value, 1e-8)
    return (
        "| Model | Approx. Param Memory (MB) |\n"
        "| --- | ---: |\n"
        f"| Expert | {expert_value:.2f} |\n"
        f"| General | {general_value:.2f} |\n"
        f"| Ratio (General / Expert) | {ratio:.2f}x |\n"
    )
