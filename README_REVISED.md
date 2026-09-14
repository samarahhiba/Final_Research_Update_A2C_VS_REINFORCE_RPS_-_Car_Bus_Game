# Policy-Gradient Learning in Zero-Sum Markov Games

**REINFORCE and A2C-style Actor-Critic in Rock-Paper-Scissors and the Car-Bus Grid**  
Author: Samarah Hiba  
Research Project: CS-298

This repository studies independent policy-gradient learning in two small zero-sum Markov games:

1. **Rock-Paper-Scissors (RPS)** — a one-state matrix game with a known uniform minimax equilibrium.
2. **Car-Bus Grid Game** — a finite-horizon 3×3 zero-sum pursuit/evasion-style grid game with spatial tile rewards and collision penalties.

The code implements:

- REINFORCE
- A2C-style episodic actor-critic
- LP minimax solver
- Tabular minimax planning
- Minimax-DQN diagnostic
- Policy averaging utilities

## Main research questions

1. Do independent REINFORCE and actor-critic updates approach the minimax equilibrium in RPS, and do they produce stable/reproducible behavior in Car-Bus?
2. When convergence fails, do the learners cycle, switch modes, or saturate into near-pure policies?
3. Does the tested A2C-style configuration reduce within-run training variability compared with REINFORCE, and does that transfer to cross-seed reliability?
4. How does Car-Bus crash-cost magnitude affect collision frequency, episode length, and return?
5. What do LP minimax planning and a short Minimax-DQN run reveal as diagnostics?

## Key findings

- **RPS:** Neither REINFORCE nor A2C reliably approaches the uniform minimax equilibrium. Policies often become seed-dependent and near-pure. Return can look acceptable while exploitability remains high.
- **Car-Bus:** The tested A2C configuration produces a smoother within-run learning signal than REINFORCE, but this does **not** imply reliable strategic behavior across seeds.
- **Post-training evaluation:** A2C and REINFORCE have similar mean collision rates, but A2C is more bimodal across seeds: some runs avoid collisions, while others collide almost immediately.
- **Crash-cost sweep:** Collision behavior is strongly seed-dependent and nonmonotone. Larger crash penalties do not automatically produce safer policies.
- **Minimax diagnostics:** Tabular minimax planning converges under the spatial-state formulation, while the short Minimax-DQN run is useful as an implementation diagnostic, not a mature baseline.

## Important interpretation note

The reported REINFORCE and A2C settings are **configuration-level comparisons**, not a critic-only causal ablation. In the current entry scripts:

- REINFORCE uses entropy coefficient `0.0`
- A2C uses entropy coefficient `0.01`

Therefore, differences in smoothness should not be attributed only to the critic.

## Reproducibility settings aligned to the paper

| Setting | Value |
|---|---:|
| RPS seeds | 0–4 |
| RPS episodes | 600 |
| Car-Bus seeds | 0–4 |
| Car-Bus episodes | 800 |
| Crash-cost sweep seeds | 0–2 |
| Crash-cost sweep episodes | 500 |
| Post-training evaluation episodes | 500 for default Car-Bus; 200 for crash-cost sweep |
| Learning rate | 0.001 |
| Discount factor | 0.95 |
| Default crash cost | 10.0 |
| Evaluation RNG seed | 10000 + training seed |
| DQN replay size / batch size | 50,000 / 128 |
| DQN target update / epsilon decay | every 500 gradient steps / 25,000 steps |

## Running the aligned scripts

Single RPS seed:

```bash
python markov_games_rl_updated/scripts/run_rps.py --seed 0 --episodes 600 --horizon 1
```

Single Car-Bus seed with post-training evaluation:

```bash
python markov_games_rl_updated/scripts/run_car_bus.py --seed 0 --episodes 800 --eval-episodes 500 --crash-cost 10
```

Optional short Minimax-DQN diagnostic:

```bash
python markov_games_rl_updated/scripts/run_car_bus.py --seed 0 --episodes 800 --run-dqn --dqn-episodes 200
```

Cross-seed report reproduction driver:

```bash
python markov_games_rl_updated/scripts/reproduce_paper.py --out outputs/reproduce_report
```

## Recommended next experiments

- Match entropy regularization between REINFORCE and A2C to isolate the critic effect.
- Expand evaluation to 10–20+ seeds.
- Evaluate frozen policies against fixed pure, mixed, planning-derived, and cross-algorithm opponents.
- Add time-to-go to the Car-Bus policy/planning/DQN state or explicitly reformulate the game as continuing.
- Execute the tabular minimax policy under the same post-training evaluation protocol.
- Report collision, episode length, state coverage, entropy, and exploitability alongside return.
