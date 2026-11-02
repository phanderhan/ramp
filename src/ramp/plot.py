import matplotlib.pyplot as plt
import numpy as np

from ramp.math import accumulated_area, ramp_steps, rate_at


def plot_ramp(
    duration: float = 4.0,
    start_rate: float = 3.0,
    end_rate: float = 4.0,
    shape_exponent: float = 1.0,
    skew: float = 0.0,
):
    t = np.linspace(0, duration, 1000)

    r = rate_at(t, duration, start_rate, end_rate, shape_exponent, skew)
    a = accumulated_area(t, duration, start_rate, end_rate, shape_exponent, skew)

    result = ramp_steps(duration, start_rate, end_rate, shape_exponent, skew)
    positions = result["positions"]

    plt.figure(figsize=(10, 4))
    plt.plot(t, r)
    complete_positions = result["boundary_positions"]
    step_rates = rate_at(
        complete_positions, duration, start_rate, end_rate, shape_exponent, skew
    )
    plt.scatter(complete_positions[:-1], step_rates[:-1], s=24, color="tab:blue",
                label="Final step times on target", zorder=3)
    plt.scatter(complete_positions[-1:], step_rates[-1:], marker="D", color="tab:orange",
                label="Boundary reference", zorder=4)
    plt.xlim(-0.05 * duration, 1.05 * duration)
    plt.legend()
    plt.title("Requested rate R(t) before closure scaling")
    plt.xlabel("Beat")
    plt.ylabel("Steps per beat")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t, a)
    step_areas = accumulated_area(
        complete_positions, duration, start_rate, end_rate, shape_exponent, skew
    )
    plt.scatter(complete_positions[:-1], step_areas[:-1], s=24, color="tab:blue",
                label="Final step times on target", zorder=3)
    plt.scatter(complete_positions[-1:], step_areas[-1:], marker="D", color="tab:orange",
                label="Boundary reference", zorder=4)
    plt.xlim(-0.05 * duration, 1.05 * duration)
    plt.legend()
    for k in range(result["original_step_count"]):
        plt.axhline(k, linewidth=0.5, alpha=0.25)
    plt.title("Requested area A(t) before closure scaling")
    plt.xlabel("Beat")
    plt.ylabel("Accumulated steps")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 1.8))
    plt.eventplot(positions, lineoffsets=1, linelengths=0.7)
    plt.plot([duration], [1], "D", color="tab:orange", label="Boundary reference (following section)")
    plt.xlim(-0.05 * duration, 1.05 * duration)
    plt.legend()
    plt.yticks([])
    plt.title(f"Generated step positions — {result['step_count']} steps")
    plt.xlabel("Beat")
    plt.grid(True)
    plt.show()
