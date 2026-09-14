# Repository alignment audit

This audit compares the current public repository summary/scripts against the research paper.

## Fixes needed

1. **README overclaims convergence.**
   - Current README says strategy averaging converged toward a mixed equilibrium and A2C generally converged more consistently.
   - Paper conclusion is more careful: neither policy-gradient method demonstrated robust equilibrium convergence or reliable cross-seed behavior.

2. **README overstates crash-cost effect.**
   - Current README says increasing crash cost encouraged conservative strategies and reduced risky actions.
   - Paper reports nonmonotone collision behavior with strong seed dependence.

3. **Car-Bus default episode count mismatch.**
   - Current `run_car_bus.py` uses 600 episodes.
   - Paper's default Car-Bus comparison uses 800 episodes.

4. **DQN epsilon-decay mismatch.**
   - Current `run_car_bus.py` uses 15,000 epsilon-decay steps.
   - Paper reports 25,000 epsilon-decay steps.

5. **Cross-seed wrapper missing.**
   - Paper reports seeds 0–4 and post-training evaluation with frozen sampled policies.
   - Current entry points are mostly single-run drivers. Add a reproducibility driver or document the wrapper explicitly.

6. **Evaluation protocol should be explicit.**
   - Paper evaluates frozen policies with evaluation seed `10000 + training_seed`.
   - Add this to code and output metrics files, not only the manuscript.

## Files included in this pack

- `README_REVISED.md` — replacement README aligned with the paper.
- `markov_games_rl_updated/scripts/run_car_bus.py` — replacement script with paper-aligned defaults and evaluation output.
- `markov_games_rl_updated/scripts/run_rps.py` — replacement script with CLI seed/lr/gamma options.
- `markov_games_rl_updated/scripts/reproduce_paper.py` — cross-seed driver scaffold for paper reproduction.
- `repository_update.patch` — patch-style instructions for the main concrete changes.
