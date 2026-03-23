#!/usr/bin/env python3
"""Fit IsoFLOPs scaling laws for model size and dataset size.

This script:
1) Loads run data from data/isoflops_curves.json.
2) Selects N_opt(C) by taking the lowest-loss run at each compute budget C.
3) Computes D_opt(C) = C / (6 * N_opt(C)).
4) Fits power laws:
     N_opt(C) = alpha_n * C^a_n
     D_opt(C) = alpha_d * C^a_d
   via linear regression in log-space.
5) Saves plots, point tables, and predictions at target compute budgets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/isoflops_curves.json"),
        help="Path to IsoFLOPs runs JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/chinchilla_isoflops"),
        help="Directory for plots and prediction artifacts.",
    )
    parser.add_argument(
        "--predict-budgets",
        type=float,
        nargs="+",
        default=[1e23, 1e24],
        help="Compute budgets (FLOPs) for extrapolated predictions.",
    )
    parser.add_argument(
        "--extrapolate-to",
        type=float,
        default=1e24,
        help="Maximum compute budget shown on plots.",
    )
    return parser.parse_args()


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (alpha, exponent) for y = alpha * x^exponent."""
    lx = np.log(x)
    ly = np.log(y)
    exponent, log_alpha = np.polyfit(lx, ly, 1)
    alpha = float(np.exp(log_alpha))
    return alpha, float(exponent)


def predict_power_law(alpha: float, exponent: float, x: np.ndarray) -> np.ndarray:
    return alpha * np.power(x, exponent)


def load_optimal_points(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = json.loads(json_path.read_text(encoding="utf-8"))
    best_by_budget: dict[float, dict] = {}
    for run in runs:
        c = float(run["compute_budget"])
        if c not in best_by_budget or float(run["final_loss"]) < float(
            best_by_budget[c]["final_loss"]
        ):
            best_by_budget[c] = run

    budgets = np.array(sorted(best_by_budget.keys()), dtype=float)
    n_opt = np.array([float(best_by_budget[c]["parameters"]) for c in budgets], dtype=float)
    d_opt = budgets / (6.0 * n_opt)
    return budgets, n_opt, d_opt


def save_points_table(path: Path, budgets: np.ndarray, values: np.ndarray, key: str) -> None:
    records = [{"compute_budget": float(c), key: float(v)} for c, v in zip(budgets, values)]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def make_log_grid(min_x: float, max_x: float, n: int = 400) -> np.ndarray:
    lo = math.log10(min_x)
    hi = math.log10(max_x)
    return np.logspace(lo, hi, n)


def plot_scaling(
    out_path: Path,
    title: str,
    y_label: str,
    budgets: np.ndarray,
    y_points: np.ndarray,
    alpha: float,
    exponent: float,
    extrapolate_to: float,
) -> None:
    x_max = max(float(np.max(budgets)), extrapolate_to)
    x_grid = make_log_grid(float(np.min(budgets)), x_max)
    y_fit = predict_power_law(alpha, exponent, x_grid)

    plt.figure(figsize=(8, 5.5))
    plt.loglog(budgets, y_points, "o", label="IsoFLOPs minima points")
    plt.loglog(
        x_grid,
        y_fit,
        "-",
        label=f"Fit: y = {alpha:.3e} * C^{exponent:.4f}",
    )
    plt.xlabel("Compute budget C (FLOPs)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    budgets, n_opt_points, d_opt_points = load_optimal_points(args.input)

    alpha_n, exp_n = fit_power_law(budgets, n_opt_points)
    alpha_d, exp_d = fit_power_law(budgets, d_opt_points)

    predict_budgets = np.array(args.predict_budgets, dtype=float)
    pred_n = predict_power_law(alpha_n, exp_n, predict_budgets)
    pred_d = predict_power_law(alpha_d, exp_d, predict_budgets)

    plot_scaling(
        out_path=args.output_dir / "model_size_scaling.png",
        title="IsoFLOPs Scaling Law: Compute-Optimal Model Size",
        y_label="N_opt(C) (parameters)",
        budgets=budgets,
        y_points=n_opt_points,
        alpha=alpha_n,
        exponent=exp_n,
        extrapolate_to=args.extrapolate_to,
    )
    plot_scaling(
        out_path=args.output_dir / "dataset_size_scaling.png",
        title="IsoFLOPs Scaling Law: Compute-Optimal Dataset Size",
        y_label="D_opt(C) (tokens)",
        budgets=budgets,
        y_points=d_opt_points,
        alpha=alpha_d,
        exponent=exp_d,
        extrapolate_to=args.extrapolate_to,
    )

    save_points_table(args.output_dir / "n_opt_points.json", budgets, n_opt_points, "n_opt")
    save_points_table(args.output_dir / "d_opt_points.json", budgets, d_opt_points, "d_opt")

    predictions = []
    for c, n, d in zip(predict_budgets, pred_n, pred_d):
        predictions.append(
            {
                "compute_budget": float(c),
                "predicted_n_opt": float(n),
                "predicted_d_opt": float(d),
            }
        )
    (args.output_dir / "predictions.json").write_text(
        json.dumps(
            {
                "model_size_fit": {"alpha": alpha_n, "exponent": exp_n},
                "dataset_size_fit": {"alpha": alpha_d, "exponent": exp_d},
                "predictions": predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = []
    lines.append(
        "Predicted compute-optimal model sizes: "
        + ", ".join(
            f"N_opt({c:.0e}) = {n:.6e} parameters" for c, n in zip(predict_budgets, pred_n)
        )
        + "."
    )
    lines.append(
        "Predicted compute-optimal dataset sizes: "
        + ", ".join(f"D_opt({c:.0e}) = {d:.6e} tokens" for c, d in zip(predict_budgets, pred_d))
        + "."
    )
    summary_text = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text, end="")
    print(f"Saved artifacts in: {args.output_dir}")


if __name__ == "__main__":
    main()
