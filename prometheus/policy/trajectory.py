"""Monotone piecewise cubic position targets with velocity/acceleration caps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySegment:
    duration: float
    coefficients: np.ndarray  # a, b, c, d in p(u) = a*u^3 + b*u^2 + c*u + d
    target: np.ndarray

    def position(self, phase: float) -> np.ndarray:
        a, b, c, d = self.coefficients
        return ((a * phase + b) * phase + c) * phase + d


def plan_acceleration_limited(
    start: np.ndarray,
    targets: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    min_interval_s: float,
) -> list[TrajectorySegment]:
    """Plan rest-to-rest per chunk, with shared interior waypoint velocities.

    Axes with infinite acceleration caps retain linear interpolation. A common
    time scale keeps all waypoints and enforces analytic derivative bounds.
    This bounds the planned target curve, not measured motor acceleration.
    """
    positions = np.vstack([start, targets]).astype(np.float64)
    delta = np.diff(positions, axis=0)
    durations = np.maximum(min_interval_s, np.max(np.abs(delta) / max_velocity, axis=1))
    slopes = delta / durations[:, None]
    velocities = np.zeros_like(positions)
    # Weighted harmonic means preserve monotonicity and avoid overshoot. At a
    # reversal, or beside a stationary segment, the common velocity is zero.
    for index in range(1, len(positions) - 1):
        left, right = slopes[index - 1], slopes[index]
        same_direction = (np.sign(left) == np.sign(right)) & (left != 0) & (right != 0)
        w1 = 2 * durations[index] + durations[index - 1]
        w2 = durations[index] + 2 * durations[index - 1]
        velocities[index, same_direction] = (w1 + w2) / (
            w1 / left[same_direction] + w2 / right[same_direction]
        )
    limited = np.isfinite(max_acceleration)
    coefficients = []
    scale = 1.0
    for index, (displacement, duration) in enumerate(zip(delta, durations)):
        c = velocities[index] * duration
        a = -2 * displacement + duration * (velocities[index] + velocities[index + 1])
        b = 3 * displacement - duration * (2 * velocities[index] + velocities[index + 1])
        a[~limited] = 0
        b[~limited] = 0
        c[~limited] = displacement[~limited]
        coefficients.append(np.stack([a, b, c, positions[index]]))
        # Velocity is quadratic: its extrema are endpoints and an interior
        # vertex. Acceleration is linear: its extrema are the endpoints.
        vertex = np.zeros_like(a)
        np.divide(-b, 3 * a, out=vertex, where=a != 0)
        vertex = np.clip(vertex, 0, 1)
        peak_velocity = np.maximum.reduce([
            np.abs(c), np.abs(3 * a + 2 * b + c),
            np.abs(3 * a * vertex**2 + 2 * b * vertex + c),
        ]) / duration
        peak_acceleration = np.maximum(np.abs(2 * b), np.abs(6 * a + 2 * b)) / duration**2
        scale = max(scale, float(np.max(peak_velocity / max_velocity)),
                    float(np.sqrt(np.max(peak_acceleration / max_acceleration))))
    return [TrajectorySegment(float(duration * scale), coef, np.asarray(target).copy())
            for duration, coef, target in zip(durations, coefficients, targets)]
