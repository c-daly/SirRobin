"""Simulation-owned time; no wall-clock reads occur in physics."""

from dataclasses import dataclass


@dataclass(slots=True)
class SimClock:
    now: float = 0.0
    step: int = 0

    def advance(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.now += float(dt)
        self.step += 1
