import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from mg.envs import RPSGame, RPSRepeatedHistoryGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json, save_state_dict_npz
from mg.policy_grad import train_reinforce, train_a2c, train_reinforce_fictitious_play
from mg.viz import save_log_csv, plot_compare, plot_metric_compare


def rps_payoff_matrix():
    return np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])


@torch.no_grad()
def policy_probs(pi, state=0, device="cpu"):
    s_t = torch.tensor([state], dtype=torch.long, device=device)
    return torch.softmax(pi.forward(s_t), dim=-1).cpu().numpy()[0]


def nashconv(p, q):
    M = rps_payoff_matrix()
    row_exploit = -float(np.min(p @ M))
    col_exploit = float(np.max(M @ q))
    return {
        "row_exploitability": row_exploit,
        "column_exploitability": col_exploit,
        "nashconv": row_exploit + col_exploit,
        "p1_l1_to_uniform": float(np.abs(p - 1.0 / 3.0).sum()),
        "p2_l1_to_uniform": float(np.abs(q - 1.0 / 3.0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Run paper-aligned RPS experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=1, help="1 = single-shot RPS. >1 uses repeated history state.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="outputs/rps")
    parser.add_argument("--fp", action="store_true", help="Also run FP-style policy averaging REINFORCE.")
    args = parser.parse_args()

    torch.set_num_threads(1)

    cfg = RunConfig(seed=args.seed, episodes=args.episodes, max_steps_per_episode=args.horizon, gamma=args.gamma, lr=args.lr, batch_size=64, device=args.device)
    out = ensure_dir(Path(args.out) / f"seed_{args.seed}_lr_{args.lr:g}_h{args.horizon}")
    save_json(out / "config.json", cfg.to_dict())

    env_factory = lambda: RPSGame() if args.horizon == 1 else RPSRepeatedHistoryGame(horizon=args.horizon)

    set_seed(cfg.seed)
    env_rein = env_factory()
    (pi1, pi2), log_rein = train_reinforce(env_rein, cfg, out / "reinforce", baseline="none", entropy_coef=0.0)
    torch.save(pi1.state_dict(), out / "reinforce_pi1.pt")
    torch.save(pi2.state_dict(), out / "reinforce_pi2.pt")
    save_state_dict_npz(out / "reinforce_pi1.npz", pi1.state_dict())
    save_state_dict_npz(out / "reinforce_pi2.npz", pi2.state_dict())
    save_log_csv(out / "reinforce_log.csv", log_rein)

    set_seed(cfg.seed)
    env_a2c = env_factory()
    (pi1a, pi2a, V), log_a2c = train_a2c(env_a2c, cfg, out / "a2c", entropy_coef=0.01, value_coef=1.0)
    torch.save(pi1a.state_dict(), out / "a2c_pi1.pt")
    torch.save(pi2a.state_dict(), out / "a2c_pi2.pt")
    torch.save(V.state_dict(), out / "a2c_value.pt")
    save_state_dict_npz(out / "a2c_pi1.npz", pi1a.state_dict())
    save_state_dict_npz(out / "a2c_pi2.npz", pi2a.state_dict())
    save_state_dict_npz(out / "a2c_value.npz", V.state_dict())
    save_log_csv(out / "a2c_log.csv", log_a2c)

    metrics = {
        "REINFORCE": nashconv(policy_probs(pi1, device=args.device), policy_probs(pi2, device=args.device)),
        "A2C": nashconv(policy_probs(pi1a, device=args.device), policy_probs(pi2a, device=args.device)),
    }
    save_json(out / "rps_final_diagnostics.json", metrics)

    if args.fp:
        set_seed(cfg.seed)
        env_fp = env_factory()
        (pi1fp, pi2fp), log_fp = train_reinforce_fictitious_play(env_fp, cfg, out / "reinforce_fp", snapshot_window=16, entropy_coef=0.0)
        torch.save(pi1fp.state_dict(), out / "reinforce_fp_pi1.pt")
        torch.save(pi2fp.state_dict(), out / "reinforce_fp_pi2.pt")
        save_state_dict_npz(out / "reinforce_fp_pi1.npz", pi1fp.state_dict())
        save_state_dict_npz(out / "reinforce_fp_pi2.npz", pi2fp.state_dict())
        save_log_csv(out / "reinforce_fp_log.csv", log_fp)

    plot_compare(log_a2c, log_rein, "A2C", "REINFORCE", out / "a2c_vs_reinforce_return.png", ma_window=50, title="RPS: A2C vs REINFORCE (P1 return)")
    plot_metric_compare(log_a2c, log_rein, "policy_loss_p1", "A2C", "REINFORCE", out / "a2c_vs_reinforce_policy_loss.png", ma_window=50, title="RPS: Policy loss (P1)")
    plot_metric_compare(log_a2c, log_rein, "value_loss", "A2C", "REINFORCE", out / "a2c_vs_reinforce_value_loss.png", ma_window=50, title="RPS: Value loss (critic)")

    print(metrics)
    print("Done. Outputs in:", out.resolve())


if __name__ == "__main__":
    main()
