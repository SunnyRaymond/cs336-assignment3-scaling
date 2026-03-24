#!/usr/bin/env python3
"""Fit scaling laws for Assignment 3 via the provided training API.

This script is designed to be launched on the cloud machine and will:
1) Query /previous_runs and /total_flops_used for the provided API key.
2) Run a budget-safe experiment plan (pilot + IsoFLOPs sweeps + validation run).
3) Fit power laws for compute-optimal model size and loss vs compute.
4) Extrapolate to the target FLOPs budget (default: 1e19).
5) Save all artifacts under results/scaling_laws/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests


ALLOWED_TRAIN_FLOPS = {
    int(1e13),
    int(3e13),
    int(6e13),
    int(1e14),
    int(3e14),
    int(6e14),
    int(1e15),
    int(3e15),
    int(6e15),
    int(1e16),
    int(3e16),
    int(6e16),
    int(1e17),
    int(3e17),
    int(6e17),
    int(1e18),
}


@dataclass(frozen=True)
class RunConfig:
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int

    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.d_model,
            self.num_layers,
            self.num_heads,
            self.batch_size,
            int(round(self.learning_rate * 1e9)),
            self.train_flops,
        )

    def non_embedding_params(self) -> int:
        # Assignment handout approximation.
        return int(12 * self.num_layers * (self.d_model**2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://hyperturing.stanford.edu:8000",
        help="Base URL for the assignment training API.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="SCALING_API_KEY",
        help="Environment variable that stores API key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scaling_laws"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=2e18,
        help="Max FLOPs budget allowed for scaling-law experiments.",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1e19,
        help="Target FLOPs budget to extrapolate to.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout per API request.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.2,
        help="Sleep between API calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and save the plan; do not call /loss.",
    )
    return parser.parse_args()


class TrainingAPIClient:
    def __init__(self, base_url: str, api_key: str, timeout_sec: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        full_params = dict(params)
        full_params["api_key"] = self.api_key
        response = requests.get(url, params=full_params, timeout=self.timeout_sec)
        payload: Any
        try:
            payload = response.json()
        except Exception:
            response.raise_for_status()
            raise RuntimeError(f"Non-JSON response from {url}")
        if response.status_code >= 400:
            raise RuntimeError(f"API error {response.status_code} on {endpoint}: {payload}")
        return payload

    def total_flops_used(self) -> float:
        try:
            out = self._get_json("/total_flops_used", {})
        except RuntimeError as exc:
            msg = str(exc)
            # Handout says 422 can happen when no queries exist yet.
            if "422" in msg:
                return 0.0
            raise
        if isinstance(out, (int, float)):
            return float(out)
        if isinstance(out, dict) and "total_flops_used" in out:
            return float(out["total_flops_used"])
        raise RuntimeError(f"Unexpected /total_flops_used response format: {out}")

    def previous_runs(self) -> list[dict[str, Any]]:
        out = self._get_json("/previous_runs", {})
        if isinstance(out, dict) and "previous_runs" in out:
            return list(out["previous_runs"])
        raise RuntimeError(f"Unexpected /previous_runs response format: {out}")

    def query_loss(self, config: RunConfig) -> dict[str, Any]:
        params = {
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "train_flops": config.train_flops,
        }
        out = self._get_json("/loss", params=params)
        if not isinstance(out, dict) or "loss" not in out:
            raise RuntimeError(f"Unexpected /loss response format: {out}")
        return out


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lx = np.log(x)
    ly = np.log(y)
    exponent, log_alpha = np.polyfit(lx, ly, 1)
    return float(np.exp(log_alpha)), float(exponent)


def predict_power_law(alpha: float, exponent: float, x: np.ndarray) -> np.ndarray:
    return alpha * np.power(x, exponent)


def pick_heads(d_model: int) -> int:
    preferred = [16, 12, 10, 8, 6, 5, 4, 3, 2]
    for h in preferred:
        if h <= 16 and h >= 2 and d_model % h == 0:
            return h
    return 2


def build_candidate_shapes() -> list[tuple[int, int, int, int]]:
    shapes: list[tuple[int, int, int, int]] = []
    for d_model in [128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]:
        for num_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
            num_heads = pick_heads(d_model)
            n_params = int(12 * num_layers * (d_model**2))
            shapes.append((n_params, d_model, num_layers, num_heads))
    shapes.sort(key=lambda x: x[0])
    return shapes


def nearest_shapes(
    target_n: float,
    candidate_shapes: list[tuple[int, int, int, int]],
    k: int,
) -> list[tuple[int, int, int, int]]:
    scored = []
    for shape in candidate_shapes:
        n_params = shape[0]
        dist = abs(math.log(n_params) - math.log(target_n))
        scored.append((dist, shape))
    scored.sort(key=lambda x: x[0])
    out: list[tuple[int, int, int, int]] = []
    used: set[tuple[int, int]] = set()
    for _, shape in scored:
        _, d_model, num_layers, _ = shape
        key = (d_model, num_layers)
        if key in used:
            continue
        used.add(key)
        out.append(shape)
        if len(out) >= k:
            break
    return out


def build_run_plan(max_budget: float) -> list[dict[str, Any]]:
    """Build a fixed plan capped to <= 2e18 intended budget usage.

    Intended default plan cost:
    - Pilot: 4 * 1e16 = 4e16
    - IsoFLOPs stage: 3 budgets * 3 sizes each:
      3 * (3e16 + 1e17 + 3e17) = 1.29e18
    - Validation: 1 * 6e17 = 6e17
    Total planned = 1.93e18
    """
    candidate_shapes = build_candidate_shapes()
    plan: list[dict[str, Any]] = []

    # Stage 0: learning-rate pilot.
    for lr in [1e-3, 7e-4, 4e-4, 2e-4]:
        plan.append(
            {
                "stage": "pilot",
                "config": asdict(
                    RunConfig(
                        d_model=512,
                        num_layers=10,
                        num_heads=8,
                        batch_size=256,
                        learning_rate=lr,
                        train_flops=int(1e16),
                    )
                ),
            }
        )

    # Stage 1: IsoFLOPs profiles with 3 model sizes per compute level.
    for c in [int(3e16), int(1e17), int(3e17)]:
        n_base = math.sqrt(c / 120.0)
        target_ns = [0.55 * n_base, 1.0 * n_base, 1.8 * n_base]
        chosen_shapes: list[tuple[int, int, int, int]] = []
        for tn in target_ns:
            for shape in nearest_shapes(tn, candidate_shapes, k=3):
                if shape not in chosen_shapes:
                    chosen_shapes.append(shape)
                    break
        chosen_shapes = chosen_shapes[:3]
        for _, d_model, num_layers, num_heads in chosen_shapes:
            plan.append(
                {
                    "stage": "isoflops",
                    "config": asdict(
                        RunConfig(
                            d_model=d_model,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            batch_size=256,
                            learning_rate=4e-4,  # Updated after pilot if pilot runs.
                            train_flops=c,
                        )
                    ),
                }
            )

    # Stage 2: one high-budget validation point.
    c_val = int(6e17)
    n_val = math.sqrt(c_val / 120.0)
    val_shape = nearest_shapes(n_val, candidate_shapes, k=1)[0]
    _, d_model, num_layers, num_heads = val_shape
    plan.append(
        {
            "stage": "validate",
            "config": asdict(
                RunConfig(
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    batch_size=256,
                    learning_rate=4e-4,
                    train_flops=c_val,
                )
            ),
        }
    )

    # If someone lowered max_budget a lot, greedily truncate by order.
    out: list[dict[str, Any]] = []
    acc = 0.0
    for item in plan:
        c = float(item["config"]["train_flops"])
        if acc + c > max_budget:
            break
        out.append(item)
        acc += c
    return out


def run_from_dict(d: dict[str, Any]) -> RunConfig:
    return RunConfig(
        d_model=int(d["d_model"]),
        num_layers=int(d["num_layers"]),
        num_heads=int(d["num_heads"]),
        batch_size=int(d["batch_size"]),
        learning_rate=float(d["learning_rate"]),
        train_flops=int(d["train_flops"]),
    )


def normalize_previous_run(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "d_model": int(raw["d_model"]),
        "num_layers": int(raw["num_layers"]),
        "num_heads": int(raw["num_heads"]),
        "batch_size": int(raw["batch_size"]),
        "learning_rate": float(raw["learning_rate"]),
        "train_flops": int(raw["train_flops"]),
        "loss": float(raw["loss"]),
    }


def plot_isoflops_profiles(path: Path, runs: list[dict[str, Any]]) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in runs:
        grouped.setdefault(int(r["train_flops"]), []).append(r)
    if not grouped:
        return
    plt.figure(figsize=(8, 5.5))
    for c in sorted(grouped.keys()):
        rs = grouped[c]
        n = np.array([float(r["non_embedding_params"]) for r in rs], dtype=float)
        l = np.array([float(r["loss"]) for r in rs], dtype=float)
        order = np.argsort(n)
        plt.plot(n[order], l[order], marker="o", label=f"C={c:.0e}")
    plt.xscale("log")
    plt.xlabel("Model size N (non-embedding parameters)")
    plt.ylabel("Final training loss")
    plt.title("IsoFLOPs Profiles from Queried Runs")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scaling_fit(
    path: Path,
    x_points: np.ndarray,
    y_points: np.ndarray,
    alpha: float,
    exponent: float,
    y_label: str,
    title: str,
    target_x: float | None = None,
    target_y: float | None = None,
) -> None:
    x_min = float(np.min(x_points))
    x_max = float(np.max(x_points))
    if target_x is not None:
        x_max = max(x_max, float(target_x))
    x_grid = np.logspace(math.log10(x_min), math.log10(x_max), 300)
    y_fit = predict_power_law(alpha, exponent, x_grid)

    plt.figure(figsize=(8, 5.5))
    plt.loglog(x_points, y_points, "o", label="Observed points")
    plt.loglog(x_grid, y_fit, "-", label=f"Fit: y = {alpha:.3e} * C^{exponent:.4f}")
    if target_x is not None and target_y is not None:
        plt.loglog([target_x], [target_y], "s", label=f"Pred @ {target_x:.0e}")
    plt.xlabel("Compute budget C (FLOPs)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {args.api_key_env} before running."
        )

    client = TrainingAPIClient(args.api_url, api_key, timeout_sec=args.request_timeout_sec)
    total_before = client.total_flops_used()
    previous = [normalize_previous_run(r) for r in client.previous_runs()]

    previous_by_key: dict[tuple[int, int, int, int, int, int], dict[str, Any]] = {}
    for r in previous:
        cfg = RunConfig(
            d_model=r["d_model"],
            num_layers=r["num_layers"],
            num_heads=r["num_heads"],
            batch_size=r["batch_size"],
            learning_rate=r["learning_rate"],
            train_flops=r["train_flops"],
        )
        previous_by_key[cfg.key()] = r

    plan = build_run_plan(max_budget=args.max_budget)
    (args.output_dir / "planned_runs.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    executed: list[dict[str, Any]] = []
    chosen_lr = 4e-4
    used_new_flops = 0.0
    remaining_budget = max(0.0, float(args.max_budget) - float(total_before))

    for item in plan:
        cfg = run_from_dict(item["config"])
        stage = str(item["stage"])
        if cfg.train_flops not in ALLOWED_TRAIN_FLOPS:
            raise RuntimeError(f"Invalid train_flops in plan: {cfg.train_flops}")

        # After pilot, overwrite later runs with selected LR.
        if stage in {"isoflops", "validate"}:
            cfg = RunConfig(
                d_model=cfg.d_model,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                batch_size=cfg.batch_size,
                learning_rate=chosen_lr,
                train_flops=cfg.train_flops,
            )

        key = cfg.key()
        record: dict[str, Any]
        if key in previous_by_key:
            p = previous_by_key[key]
            record = {
                "stage": stage,
                **asdict(cfg),
                "loss": float(p["loss"]),
                "non_embedding_params": cfg.non_embedding_params(),
                "source": "previous_runs",
            }
            executed.append(record)
            continue

        if cfg.train_flops > remaining_budget:
            continue

        if args.dry_run:
            record = {
                "stage": stage,
                **asdict(cfg),
                "loss": None,
                "non_embedding_params": cfg.non_embedding_params(),
                "source": "dry_run_planned",
            }
            executed.append(record)
            remaining_budget -= cfg.train_flops
            used_new_flops += cfg.train_flops
            continue

        out = client.query_loss(cfg)
        record = {
            "stage": stage,
            **asdict(cfg),
            "loss": float(out["loss"]),
            "non_embedding_params": cfg.non_embedding_params(),
            "total_flops_used_after_query": float(out.get("total_flops_used", 0.0)),
            "source": "api_query",
        }
        executed.append(record)
        remaining_budget -= cfg.train_flops
        used_new_flops += cfg.train_flops
        previous_by_key[key] = {
            **asdict(cfg),
            "loss": float(out["loss"]),
        }
        time.sleep(args.sleep_sec)

        # Update LR once pilot has enough finished points.
        pilot = [r for r in executed if r["stage"] == "pilot" and r["loss"] is not None]
        if len(pilot) >= 3:
            best = min(pilot, key=lambda r: float(r["loss"]))
            chosen_lr = float(best["learning_rate"])

    (args.output_dir / "executed_runs.json").write_text(
        json.dumps(executed, indent=2), encoding="utf-8"
    )

    observed = [r for r in executed if r.get("loss") is not None]
    if len(observed) < 4:
        summary = {
            "status": "insufficient_observations",
            "message": "Need at least 4 observed runs to fit scaling laws.",
            "total_flops_used_before": total_before,
            "new_flops_planned_or_used": used_new_flops,
        }
        (args.output_dir / "predictions.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    fit_runs = [r for r in observed if r["stage"] in {"isoflops", "validate"}]
    if len(fit_runs) < 3:
        fit_runs = observed

    by_budget: dict[int, list[dict[str, Any]]] = {}
    for r in fit_runs:
        by_budget.setdefault(int(r["train_flops"]), []).append(r)

    n_opt_points: list[dict[str, float]] = []
    d_opt_points: list[dict[str, float]] = []
    l_opt_points: list[dict[str, float]] = []
    for c in sorted(by_budget.keys()):
        best = min(by_budget[c], key=lambda r: float(r["loss"]))
        n_opt = float(best["non_embedding_params"])
        d_opt = float(c) / (6.0 * n_opt)
        l_opt = float(best["loss"])
        n_opt_points.append({"compute_budget": float(c), "n_opt": n_opt})
        d_opt_points.append({"compute_budget": float(c), "d_opt": d_opt})
        l_opt_points.append({"compute_budget": float(c), "loss_opt": l_opt})

    if len(n_opt_points) < 2:
        raise RuntimeError("Need at least 2 IsoFLOPs minima points to fit model-size scaling law.")
    if len(l_opt_points) < 2:
        raise RuntimeError("Need at least 2 points to fit loss scaling law.")

    c_np = np.array([p["compute_budget"] for p in n_opt_points], dtype=float)
    n_np = np.array([p["n_opt"] for p in n_opt_points], dtype=float)
    d_np = np.array([p["d_opt"] for p in d_opt_points], dtype=float)
    l_np = np.array([p["loss_opt"] for p in l_opt_points], dtype=float)

    alpha_n, exp_n = fit_power_law(c_np, n_np)
    alpha_d, exp_d = fit_power_law(c_np, d_np)
    alpha_l, exp_l = fit_power_law(c_np, l_np)

    target_c = float(args.target_flops)
    pred_n = float(predict_power_law(alpha_n, exp_n, np.array([target_c]))[0])
    pred_d = float(predict_power_law(alpha_d, exp_d, np.array([target_c]))[0])
    pred_l = float(predict_power_law(alpha_l, exp_l, np.array([target_c]))[0])

    candidate_shapes = build_candidate_shapes()
    best_shape = nearest_shapes(pred_n, candidate_shapes, k=1)[0]
    n_shape, d_model, num_layers, num_heads = best_shape
    recommended = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": 256,
        "learning_rate": chosen_lr,
        "estimated_non_embedding_params": int(n_shape),
        "estimated_tokens_at_target": float(target_c / (6.0 * n_shape)),
    }

    (args.output_dir / "n_opt_points.json").write_text(
        json.dumps(n_opt_points, indent=2), encoding="utf-8"
    )
    (args.output_dir / "d_opt_points.json").write_text(
        json.dumps(d_opt_points, indent=2), encoding="utf-8"
    )
    (args.output_dir / "loss_opt_points.json").write_text(
        json.dumps(l_opt_points, indent=2), encoding="utf-8"
    )

    fit_obj = {
        "n_opt_fit": {"alpha": alpha_n, "exponent": exp_n},
        "d_opt_fit": {"alpha": alpha_d, "exponent": exp_d},
        "loss_fit": {"alpha": alpha_l, "exponent": exp_l},
    }
    (args.output_dir / "fits.json").write_text(json.dumps(fit_obj, indent=2), encoding="utf-8")

    prediction_obj = {
        "target_flops": target_c,
        "predicted_n_opt": pred_n,
        "predicted_d_opt": pred_d,
        "predicted_loss_opt": pred_l,
        "recommended_hyperparameters": recommended,
        "total_flops_used_before": total_before,
        "new_flops_planned_or_used": used_new_flops,
        "remaining_budget_estimate": remaining_budget,
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(prediction_obj, indent=2), encoding="utf-8"
    )

    plot_isoflops_profiles(args.output_dir / "isoflops_profiles.png", fit_runs)
    plot_scaling_fit(
        path=args.output_dir / "model_size_scaling.png",
        x_points=c_np,
        y_points=n_np,
        alpha=alpha_n,
        exponent=exp_n,
        y_label="N_opt(C) (parameters)",
        title="Compute-Optimal Model Size Scaling Law",
        target_x=target_c,
        target_y=pred_n,
    )
    plot_scaling_fit(
        path=args.output_dir / "dataset_size_scaling.png",
        x_points=c_np,
        y_points=d_np,
        alpha=alpha_d,
        exponent=exp_d,
        y_label="D_opt(C) (tokens)",
        title="Compute-Optimal Dataset Size Scaling Law",
        target_x=target_c,
        target_y=pred_d,
    )
    plot_scaling_fit(
        path=args.output_dir / "loss_scaling.png",
        x_points=c_np,
        y_points=l_np,
        alpha=alpha_l,
        exponent=exp_l,
        y_label="L_opt(C) (training loss)",
        title="Compute-Optimal Loss Scaling Law",
        target_x=target_c,
        target_y=pred_l,
    )

    summary_lines = [
        f"Total FLOPs used before script: {total_before:.6e}",
        f"Additional FLOPs planned/used by this script: {used_new_flops:.6e}",
        f"Remaining budget estimate (within max_budget={args.max_budget:.6e}): {remaining_budget:.6e}",
        f"Selected pilot learning rate: {chosen_lr:.2e}",
        f"Predicted N_opt({target_c:.0e}) = {pred_n:.6e} parameters",
        f"Predicted D_opt({target_c:.0e}) = {pred_d:.6e} tokens",
        f"Predicted L_opt({target_c:.0e}) = {pred_l:.6f}",
        "Recommended hyperparameters:",
        json.dumps(recommended, indent=2),
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text, end="")
    print(f"Saved artifacts in: {args.output_dir}")


if __name__ == "__main__":
    main()

