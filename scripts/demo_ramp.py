from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ramp.math import parse_rate, ramp_steps
from ramp.plot import plot_ramp


def main():
    duration = 4.0
    start_rate_text = "3:1"
    end_rate_text = "4:1"
    shape_exponent = 1.0

    start_rate = parse_rate(start_rate_text)
    end_rate = parse_rate(end_rate_text)

    result = ramp_steps(
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        shape_exponent=shape_exponent,
    )

    print(f"Duration:         {duration:g} beats")
    print(f"Start rate:       {start_rate_text} = {start_rate:g} steps/beat")
    print(f"End rate:         {end_rate_text} = {end_rate:g} steps/beat")
    print(f"Shape exponent p: {shape_exponent:g}")
    print(f"Area:             {result['total_area']:.6f}")
    print(f"Step total:       {result['step_count']}")
    print()

    for index, position in enumerate(result["positions"], start=1):
        print(f"{index:>2}: {position:.6f}")

    plot_ramp(
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        shape_exponent=shape_exponent,
    )


if __name__ == "__main__":
    main()
