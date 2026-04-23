from __future__ import annotations
import numpy as np
from .minimax_lp import solve_minimax_cached

def planning_minimax_q(env, gamma: float = 0.95, iters: int = 10_000, tol: float = 1e-8):
    """
    envs.py defines the rules of the game, 
    and planning.py shows how those rules are 
    used to compute values through minimax updates.
    
    Tabular planning: value iteration over Q(s, a1, a2) using minimax on next state.

    ============================================================
    BIG IDEA
    ============================================================
    This computes the value of a two-player zero-sum game by repeatedly:
      1. trying every state
      2. trying every joint action
      3. stepping the environment
      4. updating Q-values using future game value

    In normal RL, you would use:
        max_a Q(s', a)

    Here, because the game is adversarial, we use:
        minimax(Q[s'])

    ============================================================
    FULL EXAMPLE
    ============================================================

    STEP 0: Current state
    --------------------------------
    Example:
        s = (0,0,2,2)

    In encoded form, this is some integer state ID.

    STEP 1: Pick a joint action
    --------------------------------
    Suppose:
        a1 = RIGHT
        a2 = LEFT

    STEP 2: Environment transition
    --------------------------------
    env.step(a1, a2) gives:

        car: (0,0) -> (1,0)
        bus: (2,2) -> (1,2)

        reward = value(1,0) - value(1,2)
               = 2 - 2
               = 0

        next state = (1,0,1,2)

    STEP 3: Compute value of next state
    --------------------------------
    Let Q[next_state] be the 4x4 game matrix at s'.

    We solve:
        v, pi = solve_minimax(Q[s'])

    Example result:
        V(s') = 1.1

    STEP 4: Bellman update
    --------------------------------
    target = r + gamma * V(s')

    Example:
        target = 0 + 0.95 * 1.1
               = 1.045

    STEP 5: Update Q entry
    --------------------------------
    Q[s, RIGHT, LEFT] = 1.045

    ============================================================
    INTUITION
    ============================================================
    env.step()     -> gives real outcome
    Q matrix       -> predicts outcomes
    minimax solver -> handles adversarial opponent

    So this file is where:
        "the environment" + "the game solver"
    get combined into a learning update.
    """
    S = env.n_states
    A = env.n_actions if hasattr(env, "n_actions") else env.n_actions
    Q = np.zeros((S, A, A), dtype=float)

    def V_of(s):
        """
        Compute the value of a state by solving the minimax game on Q[s].

        Example:
            Q[s] is a 4x4 matrix of joint-action values
            solve_minimax_cached(Q[s]) returns:
                v  = guaranteed state value
                pi = best mixed strategy for row player

        We only need v here.
        """
        M = Q[s]
        v, _ = solve_minimax_cached(M)
        return v

    for it in range(iters):
        delta = 0.0

        for s in range(S):
            for a1 in range(A):
                for a2 in range(A):

                    # To simulate from a specific encoded state, temporarily set
                    # the environment's internal state if supported.
                    if hasattr(env, "_set_state_from_id"):
                        env._set_state_from_id(s)
                    else:
                        # Stateless environments like basic RPS do not need this.
                        pass

                    # Real environment step:
                    # (state, joint action) -> (next state, reward, done)
                    s2, r1, _, done, _ = env.step(a1, a2)

                    # Multi-agent Bellman update:
                    # If terminal:
                    #   target = immediate reward
                    # Else:
                    #   target = immediate reward + discounted minimax value
                    target = r1 if done else (r1 + gamma * V_of(s2))

                    old = Q[s, a1, a2]
                    Q[s, a1, a2] = target

                    # Track max change to see when iteration converges
                    delta = max(delta, abs(old - target))

        if delta < tol:
            break

    return Q
