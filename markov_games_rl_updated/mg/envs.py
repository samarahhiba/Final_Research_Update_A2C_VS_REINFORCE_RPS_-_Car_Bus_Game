
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ACTIONS_RPS = ["R", "P", "S"]

def rps_payoff(a1: int, a2: int) -> float:
    # 0 Rock, 1 Paper, 2 Scissors
    if a1 == a2:
        return 0.0
    # Rock beats Scissors, Paper beats Rock, Scissors beats Paper
    wins = {(0,2), (1,0), (2,1)}
    return 1.0 if (a1,a2) in wins else -1.0

@dataclass
class RPSGame:
    """Stateless 2-player zero-sum Rock-Paper-Scissors."""
    n_actions: int = 3

    def reset(self):
        s = 0  # single state
        return s

    def step(self, a1: int, a2: int):
        r = rps_payoff(a1,a2)
        done = True
        s2 = 0
        info = {}
        return s2, r, -r, done, info

    @property
    def n_states(self):
        return 1


@dataclass
class RPSRepeatedHistoryGame:
    """Repeated RPS with a *history-augmented* state.

    The notes mention "fixed non-Markovian" policies. A standard way to
    implement those while still using a Markov policy network is to **augment
    the state** with a short history.

    Here the state is the previous action pair (a1_{t-1}, a2_{t-1}).
    The initial state is a dedicated start token.
    """

    horizon: int = 10
    n_actions: int = 3

    def reset(self):
        self.t = 0
        self.prev = None  # type: ignore
        return 0  # start token

    @property
    def n_states(self):
        # 1 start token + 3*3 possible previous action pairs
        return 1 + 9

    def _state_id(self):
        if self.prev is None:
            return 0
        a1, a2 = self.prev
        return 1 + a1 * 3 + a2

    def step(self, a1: int, a2: int):
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

These environments are intentionally tiny so you can visualize policies and
compare learning dynamics.
"""

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
    """
    grid_size: int = 3
    crash_cost: float = 10.0
    max_steps: int = 25
    start_car: tuple[int,int] = (0,0)
    start_bus: tuple[int,int] = (2,2)
    # value grid from the notes
    grid_values: tuple[tuple[int,int,int], tuple[int,int,int], tuple[int,int,int]] = (
        (1,2,1),
        (2,5,2),
        (1,2,1),
    )

    def reset(self):
        self.t = 0
        self.car = list(self.start_car)
        self.bus = list(self.start_bus)
        return self._state_id()

    @property
    def n_actions(self):
        return 4

    @property
    def n_states(self):
        g = self.grid_size
        return g*g*g*g

    def _state_id(self):
        g = self.grid_size
        cx, cy = self.car
        bx, by = self.bus
        return ((cx*g + cy)*g + bx)*g + by

    def _clip(self, x: int, y: int):
        g = self.grid_size
        x = min(max(x,0), g-1)
        y = min(max(y,0), g-1)
        return x,y

    def step(self, a1: int, a2: int):
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
