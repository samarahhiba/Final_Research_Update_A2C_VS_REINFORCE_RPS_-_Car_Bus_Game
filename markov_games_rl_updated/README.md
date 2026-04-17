
# Markov Games: Minimax-DQN, REINFORCE, and A2C (RPS + Car-Bus)

This mini-project implements **zero-sum Markov-game RL** for:

1. **Rock–Paper–Scissors (RPS)**: stateless, single-step, 2-player zero-sum.
2. **Car–Bus game (3×3 grid, no wrap-around)**: state is joint positions (car_x, car_y, bus_x, bus_y).
   - Action set for both players is **{U, D, L, R}** (matches the notes).
   - Per-step reward (for the car / player 1) follows the note sketch:

     \[
     r = \text{value}(\text{car square}) - \text{value}(\text{bus square}) - \mathbb{1}[\text{crash}]\cdot \text{crash\_cost}
     \]

     with value grid:

     ```
     [[1,2,1],
      [2,5,2],
      [1,2,1]]
     ```

   - Rewards are zero-sum: **bus gets -r**.

Algorithms implemented:

- **Minimax-DQN** (DQN target network + replay, backup uses **minimax value** of the next-state stage game)
- **REINFORCE** (policy gradient, baseline optional in code)
- **A2C** (actor-critic; typically smoother learning curves than raw REINFORCE)

---

## Setup

Requires: Python 3.10+.

Suggested install:

```bash
pip install numpy torch scipy matplotlib
```

---

## Run

From the project root:

### RPS
```bash
python scripts/run_rps.py --episodes 600
```

Optional (repeated RPS with history state, and FP-style averaging):

```bash
python scripts/run_rps.py --episodes 1200 --horizon 10 --fp
```

### Car–Bus
```bash
python scripts/run_car_bus.py --episodes 800
```

Optional (also train minimax DQN):

```bash
python scripts/run_car_bus.py --episodes 800 --run_dqn
```

All outputs go into `outputs/`.

---

## Output files (what each one means)

Each environment creates a folder:

- `outputs/rps/`
- `outputs/car_bus/`

Inside each folder:

### Common
- `config.json`  
  The hyperparameters used for the run (seed, gamma, learning rate, etc).

### DQN (minimax)
- `dqn_qnet.pt`  
  PyTorch weights for the learned Q network **Q(s,a1,a2)**.
- `dqn_qnet.npz`
  Same weights dumped as a NumPy `.npz` for easy inspection.
- `dqn_log.csv`  
  Per-episode log with at least:
  - `episode`
  - `return_p1` (episode return for player 1)
  - `td_loss` (mean TD loss over gradient steps in that episode)
  - `epsilon` (exploration level)

### REINFORCE
- `reinforce_pi1.pt`  
  Policy network weights for player 1.
- `reinforce_pi2.pt`  
  Policy network weights for player 2.
- `reinforce_pi1.npz`, `reinforce_pi2.npz`
  Same weights as `.npz`.
- `reinforce_log.csv`  
  Per-episode log containing:
  - `return_p1`
  - `policy_loss_p1`, `policy_loss_p2`
  - `entropy_p1`, `entropy_p2`

### REINFORCE + policy averaging (FP-style)
Only produced when running `scripts/run_rps.py --fp`.

- `reinforce_fp_pi1.pt`, `reinforce_fp_pi2.pt`
- `reinforce_fp_pi1.npz`, `reinforce_fp_pi2.npz`
- `reinforce_fp_log.csv`

### A2C
- `a2c_pi1.pt`  
  Actor weights for player 1.
- `a2c_pi2.pt`  
  Actor weights for player 2.
- `a2c_value.pt`  
  Critic/value network weights **V(s)** (for P1; P2 is -V in a zero-sum game).
- `a2c_pi1.npz`, `a2c_pi2.npz`, `a2c_value.npz`
  Same weights as `.npz`.
- `a2c_log.csv`  
  Per-episode log containing:
  - `return_p1`
  - `policy_loss_p1`, `policy_loss_p2`
  - `value_loss`
  - `entropy_p1`, `entropy_p2`

### Plots (the “photo” comparisons)

Each script writes 3 PNGs:

- `a2c_vs_reinforce_return.png`  
  Moving-average episode return (A2C is typically smoother).
- `a2c_vs_reinforce_policy_loss.png`
  Moving-average policy loss (P1).
- `a2c_vs_reinforce_value_loss.png`
  Moving-average critic loss (A2C only).

---

## Code map

- `mg/envs.py`  
  RPS + CarBus environments.
- `mg/minimax_lp.py`  
  LP-based minimax solver for stage games.
- `mg/dqn.py`  
  Minimax-DQN (target net + replay).
- `mg/policy_grad.py`  
  REINFORCE + A2C.
- `mg/viz.py`  
  CSV logger + plot utility.
- `scripts/run_rps.py`, `scripts/run_car_bus.py`  
  Entry points.

---

## Notes

- Q-network output is **linear** (no sigmoid) to avoid saturating Q values.
- Minimax value is computed by solving an LP per state (cached for speed).
