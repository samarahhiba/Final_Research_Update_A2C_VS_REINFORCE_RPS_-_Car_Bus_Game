"""Cross-seed reproduction driver for the CS-298 Markov-games paper.

This script wraps the existing single-run entry points and writes a manifest of
commands needed to reproduce the reported experiment families. It intentionally
keeps the learning implementation in `mg.policy_grad`, `mg.dqn`, and the main
entry scripts.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/reproduce_report")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Only write the command manifest.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    commands = []

    # RPS default: seeds 0-4, 600 episodes.
    for seed in range(5):
        commands.append([
            args.python, "markov_games_rl_updated/scripts/run_rps.py",
            "--seed", seed,
            "--episodes", 600,
            "--horizon", 1,
            "--out", str(out / "rps_default"),
        ])

    # RPS high learning-rate diagnostic: seed 0, 3000 episodes, lr 0.03.
    commands.append([
        args.python, "markov_games_rl_updated/scripts/run_rps.py",
        "--seed", 0,
        "--episodes", 3000,
        "--horizon", 1,
        "--lr", 0.03,
        "--out", str(out / "rps_high_lr"),
    ])

    # Car-Bus default: seeds 0-4, 800 train episodes, 500 eval episodes.
    for seed in range(5):
        commands.append([
            args.python, "markov_games_rl_updated/scripts/run_car_bus.py",
            "--seed", seed,
            "--episodes", 800,
            "--eval-episodes", 500,
            "--crash-cost", 10,
            "--out", str(out / "car_bus_default"),
        ])

    # A2C crash-cost sweep: seeds 0-2, 500 train episodes, 200 eval episodes.
    for cost in [1, 2, 5, 10, 20, 40]:
        for seed in range(3):
            commands.append([
                args.python, "markov_games_rl_updated/scripts/run_car_bus.py",
                "--seed", seed,
                "--episodes", 500,
                "--eval-episodes", 200,
                "--crash-cost", cost,
                "--out", str(out / "crash_sweep"),
            ])

    # Short DQN diagnostic.
    commands.append([
        args.python, "markov_games_rl_updated/scripts/run_car_bus.py",
        "--seed", 0,
        "--episodes", 800,
        "--eval-episodes", 500,
        "--crash-cost", 10,
        "--run-dqn",
        "--dqn-episodes", 200,
        "--out", str(out / "dqn_diagnostic"),
    ])

    with open(out / "commands_manifest.json", "w") as f:
        json.dump(commands, f, indent=2)

    if not args.dry_run:
        for cmd in commands:
            run(cmd)

    print("Manifest written to", out / "commands_manifest.json")


if __name__ == "__main__":
    main()
