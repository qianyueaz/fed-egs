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


def format_compute_savings(metrics: Dict[str, object]) -> str:
    """Format a short summary of compute savings from the routing mechanism.

    The invocation rate tells us what fraction of samples needed the general
    model.  A lower rate means the expert handled more samples on its own,
    saving the cost of running the larger general model.
    """
    invocation_rate = metrics.get("general_invocation_rate", metrics.get("invocation_rate"))
    if invocation_rate is None:
        return "Compute savings: N/A (no invocation rate recorded)\n"

    expert_only_pct = (1.0 - invocation_rate) * 100
    general_pct = invocation_rate * 100

    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples handled by expert only | {expert_only_pct:.1f}% |",
        f"| Samples requiring general model | {general_pct:.1f}% |",
    ]

    routed_acc = metrics.get("routed_accuracy")
    expert_acc = metrics.get("expert_only_accuracy")
    if routed_acc is not None and expert_acc is not None:
        improvement = (routed_acc - expert_acc) * 100
        lines.append(f"| Accuracy gain from routing | {improvement:+.2f}% |")

    return "\n".join(lines) + "\n"
