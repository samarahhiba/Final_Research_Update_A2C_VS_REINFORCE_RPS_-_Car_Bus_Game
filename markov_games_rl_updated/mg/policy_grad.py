
from __future__ import annotations
import math, random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_states, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, s_idx: torch.Tensor):
        x = self.embed(s_idx)
        x = F.relu(self.fc1(x))
        return self.logits(x)

    def dist(self, s_idx: torch.Tensor):
        logits = self.forward(s_idx)
        return torch.distributions.Categorical(logits=logits)

class ValueNet(nn.Module):
    def __init__(self, n_states: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_states, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s_idx: torch.Tensor):
        x = self.embed(s_idx)
        x = F.relu(self.fc1(x))
        return self.v(x).squeeze(-1)


class AveragedPolicy:
    """A lightweight "mixture" policy: average action probs across snapshots.

    This matches the notes' suggestion to "take average of multiple networks and
    respond to average of previous" (a practical approximation to fictitious
    play / policy averaging).
    """

    def __init__(self, snapshots: list[PolicyNet]):
        assert len(snapshots) > 0
        self.snapshots = snapshots

    @torch.no_grad()
    def dist(self, s_idx: torch.Tensor):
        # average probabilities, then sample from the mixture
        probs = None
        for pi in self.snapshots:
            p = torch.softmax(pi.forward(s_idx), dim=-1)
            probs = p if probs is None else (probs + p)
        probs = probs / len(self.snapshots)
        return torch.distributions.Categorical(probs=probs)


def clone_policy(pi: PolicyNet) -> PolicyNet:
    """Deep copy a policy network (weights only) onto the same device."""
    clone = PolicyNet(pi.embed.num_embeddings, pi.logits.out_features, hidden=pi.embed.embedding_dim)
    clone.load_state_dict({k: v.detach().clone() for k, v in pi.state_dict().items()})
    clone.to(next(pi.parameters()).device)
    clone.eval()
    return clone

def rollout_episode(env, pi1: PolicyNet, pi2: PolicyNet, device, max_steps: int):
    s = env.reset()
    traj = []
    ret1 = 0.0
    for t in range(max_steps):
        s_t = torch.tensor([s], dtype=torch.long, device=device)
        d1 = pi1.dist(s_t)
        d2 = pi2.dist(s_t)
        a1 = int(d1.sample().item())
        a2 = int(d2.sample().item())
        logp1 = d1.log_prob(torch.tensor(a1, device=device))
        logp2 = d2.log_prob(torch.tensor(a2, device=device))
        ent1 = d1.entropy()
        ent2 = d2.entropy()

        s2, r1, r2, done, _ = env.step(a1,a2)
        traj.append((s, a1, a2, float(r1), float(r2), logp1, logp2, ent1, ent2))
        ret1 += r1
        s = s2
        if done:
            break
    return traj, ret1


def rollout_episode_fixed_opponent(env, pi_train: PolicyNet, pi_fixed: PolicyNet, *,
                                  train_player: int, device, max_steps: int):
    """Rollout where one policy is trained and the other is treated as fixed.

    train_player=1 means train pi_train as player1 and keep player2 fixed.
    train_player=2 means train pi_train as player2 and keep player1 fixed.
    """
    s = env.reset()
    traj = []
    ret1 = 0.0
    for _t in range(max_steps):
        s_t = torch.tensor([s], dtype=torch.long, device=device)
        if train_player == 1:
            d1 = pi_train.dist(s_t)
            d2 = pi_fixed.dist(s_t)
        else:
            d1 = pi_fixed.dist(s_t)
            d2 = pi_train.dist(s_t)

        a1 = int(d1.sample().item())
        a2 = int(d2.sample().item())
        logp1 = d1.log_prob(torch.tensor(a1, device=device))
        logp2 = d2.log_prob(torch.tensor(a2, device=device))
        ent1 = d1.entropy(); ent2 = d2.entropy()

        s2, r1, r2, done, _ = env.step(a1, a2)
        traj.append((s, a1, a2, float(r1), float(r2), logp1, logp2, ent1, ent2))
        ret1 += r1
        s = s2
        if done:
            break
    return traj, ret1

def compute_returns(rews, gamma: float):
    G = 0.0
    out = []
    for r in reversed(rews):
        G = r + gamma*G
        out.append(G)
    out.reverse()
    return out

def train_reinforce(env, cfg, outdir, hidden=128, baseline="none", v_star_fn=None,
                    entropy_coef: float = 0.0):
    """Standard REINFORCE with optional baseline and entropy regularization.

    Logs policy loss and entropy (requested in the notes).
    baseline: 'none' | 'vstar' (requires v_star_fn(state)->float)
    """
    device = torch.device(cfg.device)
    A = env.n_actions if hasattr(env,"n_actions") else env.n_actions
    pi1 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    pi2 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    opt1 = torch.optim.Adam(pi1.parameters(), lr=cfg.lr)
    opt2 = torch.optim.Adam(pi2.parameters(), lr=cfg.lr)

    log = []
    for ep in range(cfg.episodes):
        traj, ep_ret1 = rollout_episode(env, pi1, pi2, device, cfg.max_steps_per_episode)
        rews1 = [x[3] for x in traj]
        rews2 = [x[4] for x in traj]
        G1 = compute_returns(rews1, cfg.gamma)
        G2 = compute_returns(rews2, cfg.gamma)

        # optional baseline
        b1 = []
        b2 = []
        for (s, *_rest) in traj:
            if baseline == "vstar" and v_star_fn is not None:
                v = float(v_star_fn(s))
                b1.append(v)
                b2.append(-v)
            else:
                b1.append(0.0); b2.append(0.0)

        loss1 = torch.tensor(0.0, device=device)
        loss2 = torch.tensor(0.0, device=device)
        ent1_sum = torch.tensor(0.0, device=device)
        ent2_sum = torch.tensor(0.0, device=device)
        for i,(s,a1,a2,r1,r2,logp1,logp2,ent1,ent2) in enumerate(traj):
            adv1 = (G1[i] - b1[i])
            adv2 = (G2[i] - b2[i])
            loss1 = loss1 + (-logp1 * adv1)
            loss2 = loss2 + (-logp2 * adv2)
            ent1_sum = ent1_sum + ent1
            ent2_sum = ent2_sum + ent2

        # entropy bonus (maximize entropy => subtract in loss)
        if entropy_coef != 0.0 and len(traj) > 0:
            loss1 = loss1 - entropy_coef * (ent1_sum / len(traj))
            loss2 = loss2 - entropy_coef * (ent2_sum / len(traj))

        opt1.zero_grad(); opt2.zero_grad()
        loss1.backward(); loss2.backward()
        nn.utils.clip_grad_norm_(pi1.parameters(), 10.0)
        nn.utils.clip_grad_norm_(pi2.parameters(), 10.0)
        opt1.step(); opt2.step()

        log.append({
            "episode": ep,
            "return_p1": float(ep_ret1),
            "policy_loss_p1": float(loss1.detach().cpu().item()),
            "policy_loss_p2": float(loss2.detach().cpu().item()),
            "entropy_p1": float((ent1_sum/ max(1,len(traj))).detach().cpu().item()),
            "entropy_p2": float((ent2_sum/ max(1,len(traj))).detach().cpu().item()),
        })

    return (pi1,pi2), log

def train_a2c(env, cfg, outdir, hidden=128, entropy_coef: float = 0.01, value_coef: float = 1.0):
    device = torch.device(cfg.device)
    A = env.n_actions if hasattr(env,"n_actions") else env.n_actions
    pi1 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    pi2 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    V = ValueNet(env.n_states, hidden=hidden).to(device)

    opt_pi1 = torch.optim.Adam(pi1.parameters(), lr=cfg.lr)
    opt_pi2 = torch.optim.Adam(pi2.parameters(), lr=cfg.lr)
    opt_V = torch.optim.Adam(V.parameters(), lr=cfg.lr)

    log = []
    for ep in range(cfg.episodes):
        traj, ep_ret1 = rollout_episode(env, pi1, pi2, device, cfg.max_steps_per_episode)
        rews1 = [x[3] for x in traj]
        G1 = compute_returns(rews1, cfg.gamma)

        s_list = [x[0] for x in traj]
        s_t = torch.tensor(s_list, dtype=torch.long, device=device)
        Vpred = V(s_t)

        G1_t = torch.tensor(G1, dtype=torch.float32, device=device)
        adv1 = (G1_t - Vpred).detach()

        # actor losses
        loss_pi1 = 0.0
        loss_pi2 = 0.0
        ent1_sum = torch.tensor(0.0, device=device)
        ent2_sum = torch.tensor(0.0, device=device)
        for i,(_,a1,a2,_,_,logp1,logp2,ent1,ent2) in enumerate(traj):
            # For zero-sum, use opposite advantage for player2
            loss_pi1 = loss_pi1 + (-logp1 * adv1[i])
            loss_pi2 = loss_pi2 + (-logp2 * (-adv1[i]))
            ent1_sum = ent1_sum + ent1
            ent2_sum = ent2_sum + ent2

        # critic loss
        loss_V = F.mse_loss(Vpred, G1_t)

        if entropy_coef != 0.0 and len(traj) > 0:
            loss_pi1 = loss_pi1 - entropy_coef * (ent1_sum / len(traj))
            loss_pi2 = loss_pi2 - entropy_coef * (ent2_sum / len(traj))

        opt_pi1.zero_grad(); opt_pi2.zero_grad(); opt_V.zero_grad()
        (loss_pi1 + loss_pi2 + value_coef * loss_V).backward()
        nn.utils.clip_grad_norm_(list(pi1.parameters())+list(pi2.parameters())+list(V.parameters()), 10.0)
        opt_pi1.step(); opt_pi2.step(); opt_V.step()

        log.append({
            "episode": ep,
            "return_p1": float(ep_ret1),
            "policy_loss_p1": float(loss_pi1.detach().cpu().item()),
            "policy_loss_p2": float(loss_pi2.detach().cpu().item()),
            "value_loss": float(loss_V.detach().cpu().item()),
            "entropy_p1": float((ent1_sum/ max(1,len(traj))).detach().cpu().item()),
            "entropy_p2": float((ent2_sum/ max(1,len(traj))).detach().cpu().item()),
        })

    return (pi1,pi2,V), log


def train_reinforce_fictitious_play(env, cfg, outdir, hidden=128, snapshot_window: int = 8,
                                   entropy_coef: float = 0.0):
    """REINFORCE with simple policy averaging (FP-style).

    We keep a short window of past snapshot policies for each player. When
    updating player i, we roll out against the **averaged** opponent policy and
    only update player i. This is a practical approximation of "respond to the
    average" dynamics emphasized in the notes.

    Returns (pi1,pi2), log.
    """
    device = torch.device(cfg.device)
    A = env.n_actions if hasattr(env, "n_actions") else env.n_actions
    pi1 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    pi2 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    opt1 = torch.optim.Adam(pi1.parameters(), lr=cfg.lr)
    opt2 = torch.optim.Adam(pi2.parameters(), lr=cfg.lr)

    # initialize snapshot buffers with the initial policies
    buf1 = [clone_policy(pi1)]
    buf2 = [clone_policy(pi2)]

    log = []
    for ep in range(cfg.episodes):
        # --- update player 1 vs averaged player 2 ---
        opp2 = AveragedPolicy(buf2)
        traj, ep_ret1 = rollout_episode_fixed_opponent(env, pi_train=pi1, pi_fixed=opp2,  # type: ignore
                                                       train_player=1, device=device,
                                                       max_steps=cfg.max_steps_per_episode)
        rews1 = [x[3] for x in traj]
        G1 = compute_returns(rews1, cfg.gamma)

        loss1 = torch.tensor(0.0, device=device)
        ent1_sum = torch.tensor(0.0, device=device)
        for i,(_s,a1,_a2,_r1,_r2,logp1,_logp2,ent1,_ent2) in enumerate(traj):
            loss1 = loss1 + (-logp1 * G1[i])
            ent1_sum = ent1_sum + ent1
        if entropy_coef != 0.0 and len(traj) > 0:
            loss1 = loss1 - entropy_coef * (ent1_sum / len(traj))
        opt1.zero_grad(); loss1.backward(); nn.utils.clip_grad_norm_(pi1.parameters(), 10.0); opt1.step()

        # snapshot
        buf1.append(clone_policy(pi1))
        if len(buf1) > snapshot_window:
            buf1 = buf1[-snapshot_window:]

        # --- update player 2 vs averaged player 1 ---
        opp1 = AveragedPolicy(buf1)
        traj2, _ep_ret1b = rollout_episode_fixed_opponent(env, pi_train=pi2, pi_fixed=opp1,  # type: ignore
                                                         train_player=2, device=device,
                                                         max_steps=cfg.max_steps_per_episode)
        rews2 = [x[4] for x in traj2]
        G2 = compute_returns(rews2, cfg.gamma)
        loss2 = torch.tensor(0.0, device=device)
        ent2_sum = torch.tensor(0.0, device=device)
        for i,(_s,_a1,a2,_r1,_r2,_logp1,logp2,_ent1,ent2) in enumerate(traj2):
            loss2 = loss2 + (-logp2 * G2[i])
            ent2_sum = ent2_sum + ent2
        if entropy_coef != 0.0 and len(traj2) > 0:
            loss2 = loss2 - entropy_coef * (ent2_sum / len(traj2))
        opt2.zero_grad(); loss2.backward(); nn.utils.clip_grad_norm_(pi2.parameters(), 10.0); opt2.step()

        buf2.append(clone_policy(pi2))
        if len(buf2) > snapshot_window:
            buf2 = buf2[-snapshot_window:]

        log.append({
            "episode": ep,
            "return_p1": float(ep_ret1),
            "policy_loss_p1": float(loss1.detach().cpu().item()),
            "policy_loss_p2": float(loss2.detach().cpu().item()),
            "entropy_p1": float((ent1_sum/ max(1,len(traj))).detach().cpu().item()),
            "entropy_p2": float((ent2_sum/ max(1,len(traj2))).detach().cpu().item()),
            "snapshot_window": snapshot_window,
        })

    return (pi1, pi2), log
