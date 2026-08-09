#!/usr/bin/env python3
"""Reproduce the multi-seed diagnostics used in the Markov-games research report.

Copy this file into markov_games_rl_updated/scripts/ and run it from anywhere.
It uses the project's existing mg.envs, mg.policy_grad, and mg.utils code.

Examples:
  python scripts/run_final_multiseed.py --experiment rps
  python scripts/run_final_multiseed.py --experiment car_bus
  python scripts/run_final_multiseed.py --experiment all
  python scripts/run_final_multiseed.py --experiment car_bus --crash_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

# Works when this script is inside markov_games_rl_updated/scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mg.envs import RPSGame, CarBusGame
from mg.policy_grad import train_reinforce, train_a2c
from mg.utils import RunConfig, set_seed, ensure_dir

# Single-thread execution is important for exact CPU replay of the reported runs.
torch.set_num_threads(1)

RPS_M = np.array([
    [0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0],
])


def _policy_probs(pi) -> np.ndarray:
    with torch.no_grad():
        s = torch.tensor([0], dtype=torch.long)
        return torch.softmax(pi(s), dim=-1).cpu().numpy()[0]


def _rps_diag(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    row_exploit = 0.0 - min(float(p @ RPS_M[:, j]) for j in range(3))
    col_exploit = max(float(RPS_M[i, :] @ q) for i in range(3)) - 0.0
    return {
        "row_exploit": row_exploit,
        "column_exploit": col_exploit,
        "nashconv": row_exploit + col_exploit,
        "p1_l1_uniform": float(np.abs(p - 1.0 / 3.0).sum()),
        "p2_l1_uniform": float(np.abs(q - 1.0 / 3.0).sum()),
    }


def run_rps(out: Path, seeds: list[int], episodes: int = 600) -> None:
    rows = []
    for algorithm in ("REINFORCE", "A2C"):
        for seed in seeds:
            cfg = RunConfig(
                seed=seed, episodes=episodes, max_steps_per_episode=1,
                gamma=0.95, lr=1e-3, batch_size=64, device="cpu"
            )
            set_seed(seed)
            env = RPSGame()
            if algorithm == "REINFORCE":
                (pi1, pi2), log = train_reinforce(
                    env, cfg, out / "_tmp", baseline="none", entropy_coef=0.0
                )
            else:
                (pi1, pi2, _V), log = train_a2c(
                    env, cfg, out / "_tmp", entropy_coef=0.01, value_coef=1.0
                )
            diag = _rps_diag(_policy_probs(pi1), _policy_probs(pi2))
            diag.update({
                "seed": seed,
                "algorithm": algorithm,
                "last100_return_sd": float(np.std([x["return_p1"] for x in log[-100:]], ddof=1)),
            })
            rows.append(diag)

    fields = ["seed", "algorithm", "nashconv", "row_exploit", "column_exploit",
              "p1_l1_uniform", "p2_l1_uniform", "last100_return_sd"]
    with (out / "rps_final_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary = {}
    for algorithm in ("REINFORCE", "A2C"):
        subset = [r for r in rows if r["algorithm"] == algorithm]
        summary[algorithm] = {}
        for key in fields[2:]:
            vals = [float(r[key]) for r in subset]
            summary[algorithm][key] = {
                "mean": mean(vals),
                "sample_sd": stdev(vals) if len(vals) > 1 else 0.0,
            }
    (out / "rps_summary.json").write_text(json.dumps(summary, indent=2))


def evaluate_car_bus(pi1, pi2, training_seed: int, episodes: int, crash_cost: float) -> dict[str, float]:
    # Evaluation stream used for the report: 10000 + training seed.
    set_seed(10000 + training_seed)
    env = CarBusGame(grid_size=3, crash_cost=crash_cost, max_steps=25)
    returns, collisions, lengths = [], [], []
    pi1.eval(); pi2.eval()
    with torch.no_grad():
        for _ in range(episodes):
            s = env.reset()
            ret = 0.0
            for t in range(25):
                st = torch.tensor([s], dtype=torch.long)
                a1 = int(pi1.dist(st).sample().item())
                a2 = int(pi2.dist(st).sample().item())
                s, r1, _r2, done, _ = env.step(a1, a2)
                ret += float(r1)
                if done:
                    break
            returns.append(ret)
            lengths.append(t + 1)
            collisions.append(int(env.car == env.bus))
    return {
        "mean_return": float(np.mean(returns)),
        "collision_rate": float(np.mean(collisions)),
        "mean_length": float(np.mean(lengths)),
        "return_sd": float(np.std(returns, ddof=1)),
    }


def train_car_bus(algorithm: str, seed: int, train_episodes: int, crash_cost: float):
    cfg = RunConfig(
        seed=seed, episodes=train_episodes, max_steps_per_episode=25,
        gamma=0.95, lr=1e-3, batch_size=128, device="cpu",
        replay_size=50_000, epsilon_decay_steps=25_000, target_update=500,
    )
    set_seed(seed)
    env = CarBusGame(grid_size=3, crash_cost=crash_cost, max_steps=25)
    if algorithm == "REINFORCE":
        (pi1, pi2), log = train_reinforce(
            env, cfg, Path("."), baseline="none", entropy_coef=0.0
        )
    else:
        (pi1, pi2, _V), log = train_a2c(
            env, cfg, Path("."), entropy_coef=0.01, value_coef=1.0
        )
    return pi1, pi2, log


def run_car_bus(out: Path, seeds: list[int], train_episodes: int = 800, eval_episodes: int = 500) -> None:
    rows = []
    for seed in seeds:
        for algorithm in ("A2C", "REINFORCE"):
            pi1, pi2, log = train_car_bus(algorithm, seed, train_episodes, 10.0)
            metrics = evaluate_car_bus(pi1, pi2, seed, eval_episodes, 10.0)
            metrics.update({
                "seed": seed,
                "algorithm": algorithm,
                "train_last100_return_sd": float(np.std([x["return_p1"] for x in log[-100:]], ddof=1)),
            })
            rows.append(metrics)
            print(seed, algorithm, metrics, flush=True)

    fields = ["seed", "algorithm", "mean_return", "collision_rate", "mean_length",
              "return_sd", "train_last100_return_sd"]
    with (out / "car_bus_post_train.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary = {}
    for algorithm in ("REINFORCE", "A2C"):
        subset = [r for r in rows if r["algorithm"] == algorithm]
        summary[algorithm] = {}
        for key in fields[2:]:
            vals = [float(r[key]) for r in subset]
            summary[algorithm][key] = {
                "mean": mean(vals),
                "sample_sd": stdev(vals) if len(vals) > 1 else 0.0,
            }
    (out / "car_bus_summary.json").write_text(json.dumps(summary, indent=2))


def run_crash_sweep(out: Path, seeds=(0, 1, 2), costs=(1, 2, 5, 10, 20, 40),
                    train_episodes=500, eval_episodes=200) -> None:
    rows = []
    for cost in costs:
        for seed in seeds:
            pi1, pi2, log = train_car_bus("A2C", seed, train_episodes, float(cost))
            metrics = evaluate_car_bus(pi1, pi2, seed, eval_episodes, float(cost))
            metrics.update({
                "seed": seed,
                "crash_cost": cost,
                "train_last100_collision": None,  # training logs do not store collision directly
            })
            rows.append(metrics)
            print("sweep", cost, seed, metrics, flush=True)
    fields = ["crash_cost", "seed", "mean_return", "collision_rate", "mean_length", "return_sd"]
    with (out / "crash_cost_post_train.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in fields})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["rps", "car_bus", "all"], default="all")
    p.add_argument("--outdir", default="outputs/final_multiseed")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--rps_episodes", type=int, default=600)
    p.add_argument("--car_bus_episodes", type=int, default=800)
    p.add_argument("--car_bus_eval_episodes", type=int, default=500)
    p.add_argument("--crash_sweep", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(Path(args.outdir))
    if args.experiment in ("rps", "all"):
        run_rps(out, args.seeds, args.rps_episodes)
    if args.experiment in ("car_bus", "all"):
        run_car_bus(out, args.seeds, args.car_bus_episodes, args.car_bus_eval_episodes)
    if args.crash_sweep:
        run_crash_sweep(out)
    print("Wrote reproducibility outputs to", out.resolve())


if __name__ == "__main__":
    main()
