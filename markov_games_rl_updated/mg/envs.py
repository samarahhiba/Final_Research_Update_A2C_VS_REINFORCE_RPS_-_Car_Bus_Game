
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# These are just human-readable names for the RPS actions.
# Internally we use integers:
#   0 = Rock
#   1 = Paper
#   2 = Scissors
ACTIONS_RPS = ["R", "P", "S"]

def rps_payoff(a1: int, a2: int) -> float:
    # 0 Rock, 1 Paper, 2 Scissors
    """
    Compute Player 1's reward in Rock-Paper-Scissors.

    Input:
      a1 = Player 1 action
      a2 = Player 2 action

    Output:
      +1 if Player 1 wins
       0 if tie
      -1 if Player 1 loses

    Visual idea:

        Player 1 picks one action
        Player 2 picks one action
                 ↓
          compare the pair
                 ↓
        decide win / lose / draw
                 ↓
        return reward for Player 1

    Since this is zero-sum:
      Player 2 reward = - Player 1 reward
    """
    if a1 == a2:
        return 0.0
    # These are the winning pairs for Player 1:
    # Rock beats Scissors, Paper beats Rock, Scissors beats Paper
    wins = {(0,2), (1,0), (2,1)}
    # If Player 1's pair is in the winning set, return +1.
    # Otherwise Player 1 loses and gets -1.
    return 1.0 if (a1,a2) in wins else -1.0

@dataclass
class RPSGame:
    """Stateless 2-player zero-sum Rock-Paper-Scissors."""
     """
    Stateless 2-player zero-sum Rock-Paper-Scissors.

    "Stateless" means:
    - There is no memory
    - Every round is independent
    - The current state does not depend on the past

    So the game always looks like this:

        reset() -> state 0
        choose actions
        compute reward
        episode ends immediately
    """
    n_actions: int = 3

    def reset(self):
        """
        Start a new episode.

        Since this game is stateless, there is only one state.
        We represent that single state as integer 0.

        Visual:

            start episode
                ↓
             state = 0
        """
        s = 0  # single state
        return s

    def step(self, a1: int, a2: int):
        """
        Run one step of the game.

        Visual:

            state = 0
              ↓
        Player 1 chooses a1
        Player 2 chooses a2
              ↓
          compute reward
              ↓
        episode immediately ends

        Because RPS is one-shot:
        - done = True
        - next state is still just 0
        """
        r = rps_payoff(a1,a2)
        done = True
        s2 = 0
        info = {}
        return s2, r, -r, done, info

    @property
    def n_states(self):
        """
        Number of states in this environment.

        Since it is stateless:
            number of states = 1
        """
        return 1


@dataclass
class RPSRepeatedHistoryGame:
    """Repeated RPS with a *history-augmented* state.

    The notes mention "fixed non-Markovian" policies. A  way to
    implement this while still using a Markov policy network is to **augment
    the state** with a short history.

    Here the state is the previous action pair (a1_{t-1}, a2_{t-1}).
    The initial state is a dedicated start token.
    """

    horizon: int = 10
    n_actions: int = 3

    def reset(self):
        """
        Start a new repeated-RPS episode.

        self.t = current timestep
        self.prev = previous action pair

        At the very beginning there is no previous action yet,
        so we use a special "start token" state 0.

        Visual:

            new episode
               ↓
           t = 0
        prev = None
               ↓
         return start state 0
        """
        self.t = 0
        self.prev = None  # type: ignore
        return 0  # start token

    @property
    def n_states(self):
        """
        Total states:
        - 1 special start state
        - 3*3 = 9 possible previous action pairs

        So total = 10 states

        Visual:
            start token
            + all (a1,a2) combinations
            = 10 possible states
        """
        # 1 start token + 3*3 possible previous action pairs
        return 1 + 9

    def _state_id(self):
        """
        Convert previous action pair into a discrete state ID.

        Mapping:
        - None -> 0
        - (a1, a2) -> 1 + a1*3 + a2

        Example:
          prev = (0,0) -> 1
          prev = (0,1) -> 2
          prev = (2,2) -> 9
        """
        if self.prev is None:
            return 0
        a1, a2 = self.prev
        return 1 + a1 * 3 + a2

    def step(self, a1: int, a2: int):
        """
        Run one repeated-RPS step.

        Visual:

            current state = previous action pair
                       ↓
              players choose new actions
                       ↓
                 compute reward
                       ↓
         save these actions as new history
                       ↓
             increment time step
                       ↓
          stop if horizon is reached

        This creates a tiny sequential game.
        """
        r = rps_payoff(a1, a2)
        self.prev = (a1, a2)
        self.t += 1
        done = (self.t >= self.horizon)
        return self._state_id(), r, -r, done, {}


"""Environments used in the CS298 notes.

The handwritten notes depict:

* **RPS** as a stateless zero-sum stage game.
* **Car–Bus** as a 3x3 *pursuit–evasion* Markov game (no wrap-around), with
  action set **{U, D, L, R}** for both players, and reward defined as

    reward(car) = value(car_square) - value(bus_square) + crash_term

  with the same 3x3 value grid sketched in the notes:

    [[1,2,1],
     [2,5,2],
     [1,2,1]]

 visualize policies and
compare learning dynamics.

Car–Bus is the most complex environment in this file.
It is:
- zero-sum
- spatial
- sequential
- two-player
- Markov

Visual progression of environments:

    RPSGame
      ↓
    repeated RPS with memory
      ↓
    Car-Bus pursuit-evasion on a grid

"""
# Action index -> movement direction
#
# Visual:
#   U = move up
#   D = move down
#   L = move left
#   R = move right
ACTIONS_GRID = ["U", "D", "L", "R"]

MOVE = {
    0: (0, -1),  # U
    1: (0, 1),   # D
    2: (-1, 0),  # L
    3: (1, 0),   # R
}

@dataclass
class CarBusGame:
    """Zero-sum 3x3 grid Markov game (no wrap-around).
    State = (car_x, car_y, bus_x, bus_y).

    The notes describe a pursuit–evasion flavor where the instantaneous reward
    is based on the **difference** between the value of the car's square and
    the value of the bus's square, plus an extra crash penalty.
    Rewards returned are (r_car, r_bus) with r_bus = -r_car (zero-sum).

    Zero-sum 3x3 grid Markov game (no wrap-around).

    State:
        (car_x, car_y, bus_x, bus_y)

    Big picture:
    - The car wants high-value positions
    - The bus wants to reduce the car's reward
    - If they collide, the car gets a big penalty

    Visual game loop:

        current joint positions
                ↓
        car picks action
        bus picks action
                ↓
        both move simultaneously
                ↓
        compute new reward
                ↓
      check crash or step limit
                ↓
         return next state
    """
    grid_size: int = 3
    crash_cost: float = 10.0
    max_steps: int = 25
    start_car: tuple[int,int] = (0,0)
    start_bus: tuple[int,int] = (2,2)
    # value grid from the notes
    # The value grid gives each square a score.
    #
    # Visual:
    #
    #   +---+---+---+
    #   | 1 | 2 | 1 |
    #   +---+---+---+
    #   | 2 | 5 | 2 |
    #   +---+---+---+
    #   | 1 | 2 | 1 |
    #   +---+---+---+
    #
    # The center is best.
    # This makes positioning meaningful.
    grid_values: tuple[tuple[int,int,int], tuple[int,int,int], tuple[int,int,int]] = (
        (1,2,1),
        (2,5,2),
        (1,2,1),
    )

    """
        Start a new episode.

        Car starts at top-left.
        Bus starts at bottom-right.

        Visual:

            car at (0,0)
            bus at (2,2)
                 ↓
           encode as state ID
    """
    def reset(self):
        self.t = 0
        self.car = list(self.start_car)
        self.bus = list(self.start_bus)
        return self._state_id()

    @property
    def n_actions(self):
        """
        Car and bus each have 4 actions:
            U, D, L, R
        """
        return 4

    @property
    def n_states(self):
        """
        Total number of joint states.

        Each of:
          car_x, car_y, bus_x, bus_y
        can take values in {0,1,2}

        So:
          3 * 3 * 3 * 3 = 81 states
        """
        g = self.grid_size
        return g*g*g*g

    def _state_id(self):
        """
        Convert joint positions into one integer state.

        Why do this?
        Because the RL model wants a discrete state index.

        Visual:

            (cx, cy, bx, by)
                  ↓
           unique integer in [0,80]

        So every possible pair of positions maps to a unique state ID.
        """
        g = self.grid_size
        cx, cy = self.car
        bx, by = self.bus
        return ((cx*g + cy)*g + bx)*g + by

    def _clip(self, x: int, y: int):
        """
        Keep a position inside the 3x3 grid.

        IMPORTANT:
        This game has NO wrap-around.

        So if an agent tries to move outside the board,
        it stays on the boundary instead.

        Visual:

            try to move left from x=0
                    ↓
               x would be -1
                    ↓
              clip back to x=0

        Same idea for right / up / down edges.
        """
        g = self.grid_size
        x = min(max(x,0), g-1)
        y = min(max(y,0), g-1)
        return x,y

    def step(self, a1: int, a2: int):
        """
        Execute one simultaneous step of the Car-Bus game.

        Inputs:
          a1 = car action
          a2 = bus action

        Step-by-step visual:

          Current state:
            car = (cx, cy)
            bus = (bx, by)

                 ↓ choose actions

            car chooses a1
            bus chooses a2

                 ↓ convert actions to movement

            a1 -> (dcx, dcy)
            a2 -> (dbx, dby)

                 ↓ move both at same time

            new car position = old + car delta
            new bus position = old + bus delta

                 ↓ clip positions to stay inside board

                 ↓ compute reward

            reward = value(car square) - value(bus square)

                 ↓ check crash

            if same square:
                reward -= crash_cost
                done = True

                 ↓ otherwise continue until max_steps

            return next state, rewards, done
        """
        # simultaneous moves
        dcx, dcy = MOVE[a1]
        dbx, dby = MOVE[a2]
        cx, cy = self._clip(self.car[0]+dcx, self.car[1]+dcy)
        bx, by = self._clip(self.bus[0]+dbx, self.bus[1]+dby)
        self.car = [cx,cy]
        self.bus = [bx,by]
        self.t += 1

        # reward: value(car_square) - value(bus_square) + crash term
        v_car = float(self.grid_values[cy][cx])
        v_bus = float(self.grid_values[by][bx])
        r = v_car - v_bus

        done = False
        if (cx, cy) == (bx, by):
            # crash is bad for car, good for bus
            r = r - self.crash_cost
            done = True
        elif self.t >= self.max_steps:
            done = True

        s2 = self._state_id()
        return s2, r, -r, done, {}
