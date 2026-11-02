import math
import re
from functools import lru_cache

import numpy as np


def parse_rate(rate_text: str) -> float:
    """
    Parse Scritch-style rate strings.

    Examples:
        "4:1"  -> 4.0 steps/beat
        "32:1" -> 32.0 steps/beat
        "3:2"  -> 1.5 steps/beat
        "1:32" -> 0.03125 steps/beat
    """
    text = rate_text.strip()

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(
            f'Rate must look like "4:1", "3:2", "1:32", etc. Got: {rate_text!r}'
        )

    steps = float(match.group(1))
    beats = float(match.group(2))

    if steps <= 0 or beats <= 0:
        raise ValueError("Rate values must be greater than zero.")

    if steps > 32 or beats > 32:
        raise ValueError(
            "Scritch rates should be in the range 1–32 steps per 1–32 beats."
        )

    return steps / beats


def skew_time(x, skew: float = 0.0):
    """SCRITCH time warp: negative skew transitions early, positive late."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    skew = float(np.clip(skew, -100.0, 100.0))
    if skew == 0:
        return x
    m = 0.5 + 0.45 * (skew / 100.0)
    return x / (x + (m / (1.0 - m)) * (1.0 - x))


def shaped_progress(x: np.ndarray | float, shape_exponent: float, skew: float = 0.0):
    """
    Shape normalized time x using the exponent p.

        g(x) = w(x)^p

    At zero skew:
    p = 1      linear
    p > 1      late-changing / exponential-ish
    p < 1      early-changing / logarithmic-ish
    """
    if shape_exponent <= 0:
        raise ValueError("shape_exponent must be greater than zero.")

    return np.power(skew_time(x, skew), shape_exponent)


def rate_at(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
    skew: float = 0.0,
):
    """
    Instantaneous rate curve.

        R(t) = R_start + (R_end - R_start) * g(t / D)
    """
    x = np.asarray(t) / duration
    g = shaped_progress(x, shape_exponent, skew)

    return start_rate + (end_rate - start_rate) * g


_GAUSS_RULES = {order: np.polynomial.legendre.leggauss(order) for order in (8, 16)}


class _NumericalAccumulation:
    """Adaptive Gaussian accumulation with cached panels and local inversion.

    Integrate in u, with t=D*u**power, to resolve the origin for small p.
    Panel areas are computed once. Evaluation and inversion integrate only
    inside the containing panel, never from the ramp origin for every step.
    """

    def __init__(self, duration, start_rate, end_rate, exponent, skew, rtol):
        self.duration = duration
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.exponent = exponent
        self.skew = skew
        self.power = max(1, math.ceil(2 / exponent))
        edges, areas = [0.0], []
        tolerance = rtol * duration * max(start_rate, end_rate)

        def subdivide(left, right, depth=0):
            coarse = float(self.integral(left, right, order=8))
            fine = float(self.integral(left, right))
            allowed = max(
                tolerance * (right - left),
                32 * np.finfo(float).eps * abs(fine),
            )
            if abs(fine - coarse) <= allowed:
                edges.append(right)
                areas.append(fine)
            elif depth >= 40:
                raise RuntimeError("SKEW accumulation did not converge.")
            else:
                mid = (left + right) / 2
                subdivide(left, mid, depth + 1)
                subdivide(mid, right, depth + 1)

        # Limit panel width to keep partial-panel evaluation accurate too.
        for index in range(32):
            subdivide(index / 32, (index + 1) / 32)
        self.edges = np.asarray(edges)
        self.areas = np.concatenate(([0.0], np.cumsum(areas)))

    def integral(self, left, right, order=16):
        nodes, weights = _GAUSS_RULES[order]
        left, right = np.asarray(left), np.asarray(right)
        half = (right - left) / 2
        u = (left + right)[..., None] / 2 + half[..., None] * nodes
        t = self.duration * u**self.power
        jacobian = self.duration * self.power * u ** (self.power - 1)
        values = rate_at(
            t, self.duration, self.start_rate, self.end_rate, self.exponent, self.skew
        )
        return half * np.sum(weights * values * jacobian, axis=-1)

    def area_at(self, t):
        u = np.clip(np.asarray(t) / self.duration, 0.0, 1.0) ** (1 / self.power)
        indices = np.clip(
            np.searchsorted(self.edges, u, side="right") - 1, 0, len(self.edges) - 2
        )
        return self.areas[indices] + self.integral(self.edges[indices], u)

    def times_for_areas(self, areas):
        areas = np.asarray(areas, dtype=float)
        indices = np.clip(
            np.searchsorted(self.areas, areas, side="right") - 1, 0, len(self.edges) - 2
        )
        left = self.edges[indices]
        low, high = left.copy(), self.edges[indices + 1].copy()
        targets = areas - self.areas[indices]
        for _ in range(50):
            mid = (low + high) / 2
            below = self.integral(left, mid) < targets
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        return np.where(
            areas == 0, 0.0, self.duration * ((low + high) / 2) ** self.power
        )


@lru_cache(maxsize=32)
def _numerical_accumulation(duration, start_rate, end_rate, exponent, skew, rtol=1e-12):
    if duration <= 0 or start_rate <= 0 or end_rate <= 0 or exponent <= 0:
        raise ValueError("duration, rates, and exponent must be greater than zero.")
    return _NumericalAccumulation(
        duration, start_rate, end_rate, exponent,
        float(np.clip(skew, -100, 100)), rtol,
    )


def accumulated_area(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
    skew: float = 0.0,
):
    """
    Accumulated area, analytic at zero skew and numerical otherwise:

        R(t) = R_start + (R_end - R_start) * w(t / D)^p

    A(t) = integral of R(t) from 0 to t.
    """
    if skew != 0:
        table = _numerical_accumulation(
            duration, start_rate, end_rate, shape_exponent, skew
        )
        return table.area_at(t)

    t = np.clip(np.asarray(t), 0.0, duration)
    x = t / duration
    p = shape_exponent

    return start_rate * t + (end_rate - start_rate) * duration * np.power(x, p + 1) / (
        p + 1
    )


def find_time_for_area(
    k: int,
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
    skew: float = 0.0,
) -> float:
    """
    Find t where accumulated_area(t) == k.

    Binary search works because A(t) is monotonic when rate is positive.
    """
    if k == 0:
        return 0.0
    if skew != 0:
        table = _numerical_accumulation(
            duration, start_rate, end_rate, shape_exponent, skew
        )
        return float(table.times_for_areas(k))

    low = 0.0
    high = duration

    for _ in range(80):
        mid = (low + high) / 2
        area = float(
            accumulated_area(
                mid,
                duration,
                start_rate,
                end_rate,
                shape_exponent,
            )
        )

        if area < k:
            low = mid
        else:
            high = mid

    return high


def boundary_step_data(positions, duration: float):
    """Describe generated ramp steps separately from the following section's reference.

    positions/step_count describe ramp steps excluding the reference. boundary_positions
    includes exactly one reference at duration; intervals are its differences.
    closing_interval is None when no ramp step exists. Only roundoff-level
    boundary landings are merged, without changing the generated positions.
    """
    positions = np.asarray(positions, dtype=float)
    tolerance = 8 * np.spacing(float(duration))
    if len(positions) and abs(positions[-1] - duration) <= tolerance:
        boundary_positions = positions.copy()
        boundary_positions[-1] = duration
    else:
        boundary_positions = np.append(positions, duration)

    intervals = np.diff(boundary_positions)
    return {
        "positions": positions,
        "step_count": len(positions),
        "boundary_positions": boundary_positions,
        "intervals": intervals,
        "closing_interval": float(intervals[-1]) if len(intervals) else None,
    }


def proportional_closure(positions, duration: float, natural_boundary_landing: bool):
    """Close a generated ramp without stretching an artificial remainder.

    Input positions exclude the boundary. A natural landing keeps the complete
    interval to D. Otherwise retain only original inter-step intervals and
    stretch them uniformly; the last original step becomes the reference.
    Final positions/step_count exclude that reference. Original generation is
    retained separately for inspection, not for plotting or playback.
    """
    original = np.asarray(positions, dtype=float)
    retained_intervals = np.diff(
        np.append(original, duration) if natural_boundary_landing and len(original) else original
    )
    factor = None
    if not len(original):
        status = "no_steps"
        final_steps = original
    elif natural_boundary_landing:
        status = "natural"
        factor = 1.0
        final_steps = original
    elif len(original) == 1:
        status = "single_origin_fallback"
        final_steps = original
    else:
        status = "scaled"
        factor = duration / original[-1]
        # Multiplication preserves all retained interval ratios. The terminal
        # original position is promoted to D and excluded from the ramp count.
        complete = original * factor
        complete[0] = 0.0
        complete[-1] = duration
        final_steps = complete[:-1]

    return {
        **boundary_step_data(final_steps, duration),
        "original_positions": original.copy(),
        "original_step_count": len(original),
        "original_intervals": np.diff(original),
        "retained_intervals": retained_intervals,
        "original_closing_interval": float(duration - original[-1]) if len(original) else None,
        "natural_boundary_landing": bool(natural_boundary_landing),
        "scaling_applied": status == "scaled",
        "scale_factor": factor,
        "closure_status": status,
    }


def ramp_steps(
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
    skew: float = 0.0,
):
    """
    Generate transition-shaped ACM steps, then apply proportional closure.

    The original generator output is available as original_positions. Public
    positions and intervals describe the final rhythm after closure scaling.

    Step placement rule:
        Place a step whenever accumulated area A(t) crosses an integer.

    Endpoint convention:
        Include the first step at t = 0.
        Exclude a terminal step exactly at t = duration.
    """
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    if start_rate <= 0 or end_rate <= 0:
        raise ValueError("rates must be greater than zero.")

    total_area = float(
        accumulated_area(
            duration,
            duration,
            start_rate,
            end_rate,
            shape_exponent,
            skew,
        )
    )

    epsilon = 1e-9

    # Include k = 0.
    # Exclude k = total_area when it lands exactly at the terminal endpoint.
    nearest_cycle = round(total_area)
    natural_boundary_landing = (
        nearest_cycle > 0 and abs(total_area - nearest_cycle) <= epsilon
    )
    max_k = (
        nearest_cycle - 1 if natural_boundary_landing
        else math.floor(total_area - epsilon)
    )

    if skew == 0:
        positions = [
            find_time_for_area(
                k, duration, start_rate, end_rate, shape_exponent
            )
            for k in range(max_k + 1)
        ]
    else:
        table = _numerical_accumulation(
            duration, start_rate, end_rate, shape_exponent, skew
        )
        positions = table.times_for_areas(np.arange(max_k + 1))

    return {
        "total_area": total_area,
        **proportional_closure(positions, duration, natural_boundary_landing),
    }


def interval_shape_progress(x, interval_shape_exponent: float, skew: float = 0.0):
    """
    Shape normalized time for Interval Change Mode.

        f(x) = w(x)^q

    At zero skew:
    q = 1      linear interval change
    q > 1      interval changes late
    q < 1      interval changes early
    """
    if interval_shape_exponent <= 0:
        raise ValueError("interval_shape_exponent must be greater than zero.")

    x = np.asarray(x)
    x = np.clip(x, 0.0, 1.0)

    return np.power(skew_time(x, skew), interval_shape_exponent)


def claude_curve_to_interval_exponent(curve_value: float) -> float:
    """
    Convert Claude's original CURVE value to the actual interval exponent q.

    Claude's proposal, written as exponent math:

        C = 0:
            q = 1

        C > 0:
            q = 1 + C/100

        C < 0:
            q = 1 / (1 + |C|/100)

    This is useful for comparison, but the demo should expose q directly.
    """
    curve_value = max(-100.0, min(100.0, float(curve_value)))

    if curve_value == 0:
        return 1.0

    if curve_value > 0:
        return 1.0 + curve_value / 100.0

    return 1.0 / (1.0 + abs(curve_value) / 100.0)


def interval_change_interval_at(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    interval_shape_exponent: float,
    skew: float = 0.0,
):
    """
    Interval Change Mode.

    Instead of curving rate, this curves step interval:

        I(t) = I_start + (I_end - I_start) * w(t / D)^q

    where:

        I_start = 1 / R_start
        I_end   = 1 / R_end
    """
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    if start_rate <= 0 or end_rate <= 0:
        raise ValueError("rates must be greater than zero.")

    x = np.asarray(t) / duration
    f = interval_shape_progress(x, interval_shape_exponent, skew)

    start_interval = 1.0 / start_rate
    end_interval = 1.0 / end_rate

    return start_interval + (end_interval - start_interval) * f


def interval_change_equivalent_rate_at(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    interval_shape_exponent: float,
    skew: float = 0.0,
):
    """
    Equivalent rate implied by Interval Change Mode.

    This is not the primary curve in this mode.
    It is shown only for visualization.
    """
    interval = interval_change_interval_at(
        t=t,
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        interval_shape_exponent=interval_shape_exponent,
        skew=skew,
    )

    return 1.0 / interval


def interval_change_ramp_steps(
    duration: float,
    start_rate: float,
    end_rate: float,
    interval_shape_exponent: float,
    skew: float = 0.0,
):
    """
    Generate transition-shaped ICM steps, then apply proportional closure.

    The interval walk is unchanged. original_positions records its output;
    positions and intervals describe the final rhythm after closure scaling.

    Placement rule:

        t_0 = 0
        t_next = t_current + I(t_current)

    Stop when the next step would exceed the ramp duration.
    """
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    if start_rate <= 0 or end_rate <= 0:
        raise ValueError("rates must be greater than zero.")

    epsilon = 1e-9
    max_steps = 100_000

    positions = []
    current_time = 0.0
    natural_boundary_landing = False

    while current_time < duration - epsilon:
        interval = float(
            interval_change_interval_at(
                t=current_time,
                duration=duration,
                start_rate=start_rate,
                end_rate=end_rate,
                interval_shape_exponent=interval_shape_exponent,
                skew=skew,
            )
        )

        if not np.isfinite(interval) or interval <= 0:
            break

        # If even the first interval cannot fit, this mode produces zero steps.
        if not positions and interval > duration + epsilon:
            break

        positions.append(current_time)

        next_time = current_time + interval

        if next_time >= duration - epsilon:
            natural_boundary_landing = abs(next_time - duration) <= epsilon
            break

        current_time = next_time

        if len(positions) >= max_steps:
            raise RuntimeError("Too many generated steps. Check ramp settings.")

    return {
        "mode": "Interval Change Mode",
        "interval_shape_exponent": interval_shape_exponent,
        **proportional_closure(positions, duration, natural_boundary_landing),
        "start_interval": 1.0 / start_rate,
        "end_interval": 1.0 / end_rate,
    }
