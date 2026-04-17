import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import torch

from mg.envs import CarBusGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json, save_state_dict_npz
from mg.policy_grad import train_reinforce, train_a2c
from mg.dqn import train_dqn_minimax
from mg.viz import save_log_csv, plot_compare, plot_metric_compare


def add_set_state(env: CarBusGame):
    """Adds a helper used by planning_minimax_q if you want planning later."""
    def _set_state_from_id(sid: int):
        g = env.grid_size
        by = sid % g
        sid //= g
        bx = sid % g
        sid //= g
        cy = sid % g
        cx = sid // g
        env.t = 0
        env.car = [int(cx), int(cy)]
        env.bus = [int(bx), int(by)]
    env._set_state_from_id = _set_state_from_id
    return env


def main():
    run_dqn = bool(int(os.environ.get("RUN_DQN", "0")))

    cfg = RunConfig(
        seed=0,
        episodes=600,
        max_steps_per_episode=25,
        gamma=0.95,
        lr=1e-3,
        batch_size=128,
        replay_size=50_000,
        target_update=500,
        epsilon_decay_steps=15_000,
        device="cpu",
    )
    set_seed(cfg.seed)

    # matches  current mg.envs.CarBusGame
    env = add_set_state(
        CarBusGame(
            grid_size=3,
            crash_cost=10.0,
            max_steps=25,
        )
    )

    out = ensure_dir(Path("outputs") / "car_bus")
    save_json(out / "config.json", cfg.to_dict())

    # --- Minimax DQN (optional) ---
    if run_dqn:
        qnet, log_dqn = train_dqn_minimax(env, cfg, out / "dqn")
        torch.save(qnet.state_dict(), out / "dqn_qnet.pt")
        save_state_dict_npz(out / "dqn_qnet.npz", qnet.state_dict())
        save_log_csv(out / "dqn_log.csv", log_dqn)

    # --- REINFORCE ---
    (pi1, pi2), log_rein = train_reinforce(
        env, cfg, out / "reinforce", baseline="none", entropy_coef=0.0
    )
    torch.save(pi1.state_dict(), out / "reinforce_pi1.pt")
    torch.save(pi2.state_dict(), out / "reinforce_pi2.pt")
    save_state_dict_npz(out / "reinforce_pi1.npz", pi1.state_dict())
    save_state_dict_npz(out / "reinforce_pi2.npz", pi2.state_dict())
    save_log_csv(out / "reinforce_log.csv", log_rein)

    # --- A2C ---
    (pi1a, pi2a, V), log_a2c = train_a2c(
        env, cfg, out / "a2c", entropy_coef=0.01, value_coef=1.0
    )
    torch.save(pi1a.state_dict(), out / "a2c_pi1.pt")
    torch.save(pi2a.state_dict(), out / "a2c_pi2.pt")
    torch.save(V.state_dict(), out / "a2c_value.pt")
    save_state_dict_npz(out / "a2c_pi1.npz", pi1a.state_dict())
    save_state_dict_npz(out / "a2c_pi2.npz", pi2a.state_dict())
    save_state_dict_npz(out / "a2c_value.npz", V.state_dict())
    save_log_csv(out / "a2c_log.csv", log_a2c)

    # --- Plots ---
    plot_compare(
        log_a2c,
        log_rein,
        "A2C",
        "REINFORCE",
        out / "a2c_vs_reinforce_return.png",
        ma_window=50,
        title="Car-Bus: A2C vs REINFORCE (P1 return)",
    )
    plot_metric_compare(
        log_a2c,
        log_rein,
        "policy_loss_p1",
        "A2C",
        "REINFORCE",
        out / "a2c_vs_reinforce_policy_loss.png",
        ma_window=50,
        title="Car-Bus: Policy loss (P1)",
    )
    plot_metric_compare(
        log_a2c,
        log_rein,
        "value_loss",
        "A2C",
        "REINFORCE",
        out / "a2c_vs_reinforce_value_loss.png",
        ma_window=50,
        title="Car-Bus: Value loss (critic)",
    )

    print("Done. Outputs in:", out.resolve())
    print("RUN_DQN =", int(run_dqn))


if __name__ == "__main__":
    main()
