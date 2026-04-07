import json
from pathlib import Path
from typing import Dict, List

from fedegs.federated.common import RoundMetrics


def save_metrics(output_dir: str, history: List[RoundMetrics], results: Dict[str, object]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    serializable_history = [
        {
            "round": item.round_idx,
            "avg_client_loss": item.avg_client_loss,
            "routed_accuracy": item.routed_accuracy,
            "hard_accuracy": item.hard_accuracy,
            "invocation_rate": item.invocation_rate,
            "local_accuracy": item.local_accuracy,
            "compute_savings": item.compute_savings,
        }
        for item in history
    ]
    (path / "round_metrics.json").write_text(json.dumps(serializable_history, indent=2), encoding="utf-8")
    (path / "experiment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def format_memory_table(memory_metrics: Dict[str, float]) -> str:
    if not memory_metrics or "expert" not in memory_metrics or "general" not in memory_metrics:
        return "| Model | Approx. Param Memory (MB) |\n| --- | ---: |\n| N/A | N/A |\n"
    expert = memory_metrics["expert"]
    general = memory_metrics["general"]
    ratio = general / max(expert, 1e-8)
    return (
        "| Model | Approx. Param Memory (MB) |\n"
        "| --- | ---: |\n"
        f"| Expert (0.25x) | {expert:.2f} |\n"
        f"| General (1.0x) | {general:.2f} |\n"
        f"| Ratio (General / Expert) | {ratio:.2f}x |\n"
    )
