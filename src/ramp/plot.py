import matplotlib.pyplot as plt
import numpy as np

from ramp.math import accumulated_area, ramp_steps, rate_at


def plot_ramp(
    duration: float = 4.0,
    start_rate: float = 3.0,
    end_rate: float = 4.0,
    shape_exponent: float = 1.0,
):
    t = np.linspace(0, duration, 1000)

    r = rate_at(t, duration, start_rate, end_rate, shape_exponent)
    a = accumulated_area(t, duration, start_rate, end_rate, shape_exponent)

    result = ramp_steps(duration, start_rate, end_rate, shape_exponent)
    positions = result["positions"]

    plt.figure(figsize=(10, 4))
    plt.plot(t, r)
    plt.title("Instantaneous rate R(t)")
    plt.xlabel("Beat")
    plt.ylabel("Steps per beat")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t, a)
    for k in range(result["step_count"]):
        plt.axhline(k, linewidth=0.5, alpha=0.25)
    plt.title("Accumulated area A(t)")
    plt.xlabel("Beat")
    plt.ylabel("Accumulated steps")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 1.8))
    plt.eventplot(positions, lineoffsets=1, linelengths=0.7)
    plt.xlim(0, duration)
    plt.yticks([])
    plt.title(f"Generated step positions — {result['step_count']} steps")
    plt.xlabel("Beat")
    plt.grid(True)
    plt.show()
