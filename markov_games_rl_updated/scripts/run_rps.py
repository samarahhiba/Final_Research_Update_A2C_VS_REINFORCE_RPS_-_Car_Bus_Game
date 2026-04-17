import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import torch

from mg.envs import RPSGame, RPSRepeatedHistoryGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json, save_state_dict_npz
from mg.policy_grad import train_reinforce, train_a2c, train_reinforce_fictitious_play
from mg.viz import save_log_csv, plot_compare, plot_metric_compare


def main():
    ap = argparse.ArgumentParser(description="Run RPS experiments (REINFORCE vs A2C, with optional FP averaging).")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=1, help="Episode length. 1 = single-shot RPS. >1 uses repeated RPS.")
    ap.add_argument("--fp", action="store_true", help="Also run FP-style policy averaging REINFORCE.")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = RunConfig(seed=0, episodes=args.episodes, max_steps_per_episode=args.horizon, gamma=0.95,
                    lr=1e-3, batch_size=64, device=args.device)
    set_seed(cfg.seed)

    env = RPSGame() if args.horizon == 1 else RPSRepeatedHistoryGame(horizon=args.horizon)

    out = ensure_dir(Path("outputs") / "rps")
    save_json(out / "config.json", cfg.to_dict())

    # --- REINFORCE ---
    (pi1, pi2), log_rein = train_reinforce(env, cfg, out / "reinforce", baseline="none", entropy_coef=0.0)
    torch.save(pi1.state_dict(), out / "reinforce_pi1.pt")
    torch.save(pi2.state_dict(), out / "reinforce_pi2.pt")
    save_state_dict_npz(out / "reinforce_pi1.npz", pi1.state_dict())
    save_state_dict_npz(out / "reinforce_pi2.npz", pi2.state_dict())
    save_log_csv(out / "reinforce_log.csv", log_rein)

    # --- A2C ---
    (pi1a, pi2a, V), log_a2c = train_a2c(env, cfg, out / "a2c", entropy_coef=0.01, value_coef=1.0)
    torch.save(pi1a.state_dict(), out / "a2c_pi1.pt")
    torch.save(pi2a.state_dict(), out / "a2c_pi2.pt")
    torch.save(V.state_dict(), out / "a2c_value.pt")
    save_state_dict_npz(out / "a2c_pi1.npz", pi1a.state_dict())
    save_state_dict_npz(out / "a2c_pi2.npz", pi2a.state_dict())
    save_state_dict_npz(out / "a2c_value.npz", V.state_dict())
    save_log_csv(out / "a2c_log.csv", log_a2c)

    # --- Optional: FP averaging (REINFORCE) ---
    if args.fp:
        (pi1fp, pi2fp), log_fp = train_reinforce_fictitious_play(env, cfg, out / "reinforce_fp",
                                                                 snapshot_window=16, entropy_coef=0.0)
        torch.save(pi1fp.state_dict(), out / "reinforce_fp_pi1.pt")
        torch.save(pi2fp.state_dict(), out / "reinforce_fp_pi2.pt")
        save_state_dict_npz(out / "reinforce_fp_pi1.npz", pi1fp.state_dict())
        save_state_dict_npz(out / "reinforce_fp_pi2.npz", pi2fp.state_dict())
        save_log_csv(out / "reinforce_fp_log.csv", log_fp)

    # --- Plots (requested "photos") ---
    plot_compare(log_a2c, log_rein, "A2C", "REINFORCE", out / "a2c_vs_reinforce_return.png",
                 ma_window=50, title="RPS: A2C vs REINFORCE (P1 return)")
    plot_metric_compare(log_a2c, log_rein, "policy_loss_p1", "A2C", "REINFORCE",
                        out / "a2c_vs_reinforce_policy_loss.png", ma_window=50,
                        title="RPS: Policy loss (P1)")
    plot_metric_compare(log_a2c, log_rein, "value_loss", "A2C", "REINFORCE",
                        out / "a2c_vs_reinforce_value_loss.png", ma_window=50,
                        title="RPS: Value loss (critic)")

    print("Done. Outputs in:", out.resolve())


if __name__ == "__main__":
    main()
