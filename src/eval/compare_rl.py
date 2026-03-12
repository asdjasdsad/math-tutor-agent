"""Compare PPO and GRPO metrics and generate a Markdown report with plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PPO and GRPO runs.")
    parser.add_argument("--ppo-metrics", required=True)
    parser.add_argument("--grpo-metrics", required=True)
    parser.add_argument("--report-path", default="reports/ppo_vs_grpo_report.md")
    return parser.parse_args()


def _load_metric_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(load_jsonl(path))
    if "step" in frame.columns:
        frame = frame[frame["step"].notna()].copy()
    return frame


def _plot_metric(ppo_df: pd.DataFrame, grpo_df: pd.DataFrame, metric: str, report_dir: Path) -> Path | None:
    if metric not in ppo_df.columns or metric not in grpo_df.columns:
        return None
    figure_path = report_dir / f"{metric}.png"
    plt.figure(figsize=(8, 4))
    plt.plot(ppo_df["step"], ppo_df[metric], label="PPO")
    plt.plot(grpo_df["step"], grpo_df[metric], label="GRPO")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.title(f"PPO vs GRPO: {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def _latest_metric(frame: pd.DataFrame, metric: str) -> float | None:
    if metric not in frame.columns or frame.empty:
        return None
    value = frame[metric].dropna()
    if value.empty:
        return None
    return float(value.iloc[-1])


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path)
    report_dir = ensure_dir(report_path.parent)

    ppo_df = _load_metric_frame(args.ppo_metrics)
    grpo_df = _load_metric_frame(args.grpo_metrics)
    ppo_df.to_csv(report_dir / "ppo_metrics.csv", index=False)
    grpo_df.to_csv(report_dir / "grpo_metrics.csv", index=False)

    metrics_to_plot = ["mean_reward", "tokens_per_sec", "samples_per_sec", "peak_memory_gb"]
    plot_paths = [path for metric in metrics_to_plot if (path := _plot_metric(ppo_df, grpo_df, metric, report_dir))]

    table_metrics = [
        "mean_reward",
        "mean_correctness_reward",
        "mean_rlaif_reward",
        "mean_format_reward",
        "tokens_per_sec",
        "samples_per_sec",
        "peak_memory_gb",
        "elapsed_sec",
    ]

    lines = [
        "# PPO vs GRPO Report",
        "",
        "## Summary",
        "",
        "| Metric | PPO | GRPO |",
        "| --- | ---: | ---: |",
    ]
    for metric in table_metrics:
        ppo_value = _latest_metric(ppo_df, metric)
        grpo_value = _latest_metric(grpo_df, metric)
        if ppo_value is None and grpo_value is None:
            continue
        lines.append(
            f"| {metric} | {ppo_value if ppo_value is not None else 'N/A'} | {grpo_value if grpo_value is not None else 'N/A'} |"
        )

    if plot_paths:
        lines.extend(["", "## Curves", ""])
        for plot_path in plot_paths:
            lines.append(f"![{plot_path.stem}]({plot_path.name})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
