import math
import re

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


def shaped_progress(x: np.ndarray | float, shape_exponent: float):
    """
    Shape normalized time x using the exponent p.

        g(x) = x^p

    p = 1      linear
    p > 1      late-changing / exponential-ish
    p < 1      early-changing / logarithmic-ish
    """
    if shape_exponent <= 0:
        raise ValueError("shape_exponent must be greater than zero.")

    return np.power(x, shape_exponent)


def rate_at(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
):
    """
    Instantaneous rate curve.

        R(t) = R_start + (R_end - R_start) * g(t / D)
    """
    x = np.asarray(t) / duration
    g = shaped_progress(x, shape_exponent)

    return start_rate + (end_rate - start_rate) * g


def accumulated_area(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
):
    """
    Closed form for accumulated area under:

        R(t) = R_start + (R_end - R_start) * (t / D)^p

    A(t) = integral of R(t) from 0 to t.
    """
    t = np.asarray(t)
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
) -> float:
    """
    Find t where accumulated_area(t) == k.

    Binary search works because A(t) is monotonic when rate is positive.
    """
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


def ramp_steps(
    duration: float,
    start_rate: float,
    end_rate: float,
    shape_exponent: float,
):
    """
    Generate step positions.

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
        )
    )

    epsilon = 1e-9

    # Include k = 0.
    # Exclude k = total_area when it lands exactly at the terminal endpoint.
    max_k = math.floor(total_area - epsilon)

    positions = [
        find_time_for_area(
            k,
            duration,
            start_rate,
            end_rate,
            shape_exponent,
        )
        for k in range(max_k + 1)
    ]

    intervals = np.diff(positions)

    return {
        "total_area": total_area,
        "step_count": len(positions),
        "positions": np.array(positions),
        "intervals": intervals,
    }


def claude_curve_progress(x, curve_value: float):
    """
    Claude/Jon interval-mode curve function, implemented as written.

    C = 0:
        f(x) = x

    C > 0:
        f(x) = x^(1 + C/100)

    C < 0:
        f(x) = x^(1 / (1 + |C|/100))

    Note:
        These formulas are implemented literally from Claude's proposal.
        Their behavior is intentionally shown in the demo for comparison.
    """
    curve_value = max(-100.0, min(100.0, float(curve_value)))
    x = np.asarray(x)
    x = np.clip(x, 0.0, 1.0)

    if curve_value == 0:
        return x

    if curve_value > 0:
        exponent = 1.0 + curve_value / 100.0
    else:
        exponent = 1.0 / (1.0 + abs(curve_value) / 100.0)

    return np.power(x, exponent)


def claude_interval_at(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    curve_value: float,
):
    """
    Claude/Jon interval-mode model.

    Instead of curving rate, this curves interval:

        I(t) = I_start + (I_end - I_start) * f(t / D, C)

    where:

        I_start = 1 / R_start
        I_end   = 1 / R_end
    """
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    if start_rate <= 0 or end_rate <= 0:
        raise ValueError("rates must be greater than zero.")

    x = np.asarray(t) / duration
    f = claude_curve_progress(x, curve_value)

    start_interval = 1.0 / start_rate
    end_interval = 1.0 / end_rate

    return start_interval + (end_interval - start_interval) * f


def claude_equivalent_rate_at(
    t,
    duration: float,
    start_rate: float,
    end_rate: float,
    curve_value: float,
):
    """
    Equivalent rate implied by Claude's interval model.

    This is not the primary curve in Claude's model.
    It is shown only for visualization.
    """
    interval = claude_interval_at(
        t=t,
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        curve_value=curve_value,
    )

    return 1.0 / interval


def claude_interval_ramp_steps(
    duration: float,
    start_rate: float,
    end_rate: float,
    curve_value: float,
):
    """
    Generate step positions using Claude's interval-interpolation model.

    Placement rule:

        t_0 = 0
        t_next = t_current + I(t_current)

    Stop when the next step would exceed the ramp duration.

    Endpoint convention for this demo:
        - include t = 0 only if at least one interval can fit
        - exclude a step exactly at t = duration
        - if the first interval is longer than the duration, return zero steps

    That preserves Claude's idea that very slow rates can produce zero steps.
    """
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    if start_rate <= 0 or end_rate <= 0:
        raise ValueError("rates must be greater than zero.")

    epsilon = 1e-9
    max_steps = 100_000

    positions = []
    current_time = 0.0

    while current_time < duration - epsilon:
        interval = float(
            claude_interval_at(
                t=current_time,
                duration=duration,
                start_rate=start_rate,
                end_rate=end_rate,
                curve_value=curve_value,
            )
        )

        if not np.isfinite(interval) or interval <= 0:
            break

        # If even the first interval cannot fit, the model produces zero steps.
        if not positions and interval > duration + epsilon:
            break

        positions.append(current_time)

        next_time = current_time + interval

        if next_time >= duration - epsilon:
            break

        current_time = next_time

        if len(positions) >= max_steps:
            raise RuntimeError("Too many generated steps. Check ramp settings.")

    positions = np.array(positions)
    intervals = np.diff(positions)

    return {
        "mode": "Interval Change Mode",
        "curve_value": curve_value,
        "step_count": len(positions),
        "positions": positions,
        "intervals": intervals,
        "start_interval": 1.0 / start_rate,
        "end_interval": 1.0 / end_rate,
    }
