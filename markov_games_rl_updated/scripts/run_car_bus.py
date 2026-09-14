import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from mg.envs import CarBusGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json, save_state_dict_npz
from mg.policy_grad import train_reinforce, train_a2c
from mg.dqn import train_dqn_minimax, select_actions_from_Q
from mg.viz import save_log_csv, plot_compare, plot_metric_compare


def add_set_state(env: CarBusGame):
    """Helper used by tabular/DQN minimax code to decode a spatial state id."""
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


def make_env(crash_cost: float, max_steps: int) -> CarBusGame:
    return add_set_state(CarBusGame(grid_size=3, crash_cost=crash_cost, max_steps=max_steps))


@torch.no_grad()
def evaluate_policy_pair(pi1, pi2, *, crash_cost: float, max_steps: int, episodes: int, seed: int, device: str = "cpu"):
    """Evaluate frozen sampled policies using the paper protocol."""
    set_seed(seed)
    env = make_env(crash_cost, max_steps)
    returns, collisions, lengths = [], [], []
    for _ in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        collided = 0
        steps = 0
        for _t in range(max_steps):
            s_t = torch.tensor([s], dtype=torch.long, device=device)
            a1 = int(pi1.dist(s_t).sample().item())
            a2 = int(pi2.dist(s_t).sample().item())
            s, r1, _r2, done, _info = env.step(a1, a2)
            ep_ret += float(r1)
            steps += 1
            if env.car == env.bus:
                collided = 1
            if done:
                break
        returns.append(ep_ret)
        collisions.append(collided)
        lengths.append(steps)
    return {
        "eval_episodes": episodes,
        "eval_seed": seed,
        "mean_return": float(np.mean(returns)),
        "return_sd": float(np.std(returns, ddof=1)) if episodes > 1 else 0.0,
        "collision_rate": float(np.mean(collisions)),
        "mean_length": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run paper-aligned Car-Bus experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=800, help="Paper default for Car-Bus policy-gradient runs.")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--crash-cost", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="outputs/car_bus")
    parser.add_argument("--run-dqn", action="store_true")
    parser.add_argument("--dqn-episodes", type=int, default=200, help="Short diagnostic run used in the paper.")
    args = parser.parse_args()

    torch.set_num_threads(1)

    cfg = RunConfig(
        seed=args.seed,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=128,
        replay_size=50_000,
        target_update=500,
        epsilon_decay_steps=25_000,
        device=args.device,
    )

    out = ensure_dir(Path(args.out) / f"seed_{args.seed}_crash_{args.crash_cost:g}")
    save_json(out / "config.json", cfg.to_dict())

    set_seed(cfg.seed)
    env_rein = make_env(args.crash_cost, args.max_steps)
    (pi1, pi2), log_rein = train_reinforce(env_rein, cfg, out / "reinforce", baseline="none", entropy_coef=0.0)
    torch.save(pi1.state_dict(), out / "reinforce_pi1.pt")
    torch.save(pi2.state_dict(), out / "reinforce_pi2.pt")
    save_state_dict_npz(out / "reinforce_pi1.npz", pi1.state_dict())
    save_state_dict_npz(out / "reinforce_pi2.npz", pi2.state_dict())
    save_log_csv(out / "reinforce_log.csv", log_rein)
    eval_rein = evaluate_policy_pair(
        pi1, pi2,
        crash_cost=args.crash_cost,
        max_steps=args.max_steps,
        episodes=args.eval_episodes,
        seed=10000 + args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)
    env_a2c = make_env(args.crash_cost, args.max_steps)
    (pi1a, pi2a, V), log_a2c = train_a2c(env_a2c, cfg, out / "a2c", entropy_coef=0.01, value_coef=1.0)
    torch.save(pi1a.state_dict(), out / "a2c_pi1.pt")
    torch.save(pi2a.state_dict(), out / "a2c_pi2.pt")
    torch.save(V.state_dict(), out / "a2c_value.pt")
    save_state_dict_npz(out / "a2c_pi1.npz", pi1a.state_dict())
    save_state_dict_npz(out / "a2c_pi2.npz", pi2a.state_dict())
    save_state_dict_npz(out / "a2c_value.npz", V.state_dict())
    save_log_csv(out / "a2c_log.csv", log_a2c)
    eval_a2c = evaluate_policy_pair(
        pi1a, pi2a,
        crash_cost=args.crash_cost,
        max_steps=args.max_steps,
        episodes=args.eval_episodes,
        seed=10000 + args.seed,
        device=args.device,
    )

    save_json(out / "post_training_eval.json", {"REINFORCE": eval_rein, "A2C": eval_a2c})

    if args.run_dqn:
        dqn_cfg = RunConfig(**{**cfg.to_dict(), "episodes": args.dqn_episodes})
        set_seed(cfg.seed)
        env_dqn = make_env(args.crash_cost, args.max_steps)
        qnet, log_dqn = train_dqn_minimax(env_dqn, dqn_cfg, out / "dqn")
        torch.save(qnet.state_dict(), out / "dqn_qnet.pt")
        save_state_dict_npz(out / "dqn_qnet.npz", qnet.state_dict())
        save_log_csv(out / "dqn_log.csv", log_dqn)

    plot_compare(log_a2c, log_rein, "A2C", "REINFORCE", out / "a2c_vs_reinforce_return.png", ma_window=50, title="Car-Bus: A2C vs REINFORCE (P1 return)")
    plot_metric_compare(log_a2c, log_rein, "policy_loss_p1", "A2C", "REINFORCE", out / "a2c_vs_reinforce_policy_loss.png", ma_window=50, title="Car-Bus: Policy loss (P1)")
    plot_metric_compare(log_a2c, log_rein, "value_loss", "A2C", "REINFORCE", out / "a2c_vs_reinforce_value_loss.png", ma_window=50, title="Car-Bus: Value loss (critic)")

    print(json.dumps({"output": str(out.resolve()), "REINFORCE": eval_rein, "A2C": eval_a2c}, indent=2))


if __name__ == "__main__":
    main()
