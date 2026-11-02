"""Run with: MPLCONFIGDIR=/tmp/ramp-mpl .venv/bin/python -m unittest discover -s tests."""
import ast
import io
from pathlib import Path
import sys
import unittest
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ramp.math import boundary_step_data, interval_change_ramp_steps, ramp_steps


def app_helpers():
    # Load actual chart/audio functions without executing Streamlit's page UI.
    tree = ast.parse((ROOT / "apps/ramp_app.py").read_text())
    functions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    import ramp.math as ramp_math
    namespace = dict(vars(ramp_math), plt=plt, io=io, wave=wave)
    exec(compile(functions, "apps/ramp_app.py", "exec"), namespace)
    return namespace


class BoundaryTests(unittest.TestCase):
    cases = [
        (4, 3, 4, 1),       # acceleration, ACM exact area; ICM remainder
        (4, 4, 3, 1),       # deceleration
        (2.25, 1, 2, 2),    # nonzero remainder
        (2.25, 2, 1, 0.5),
        (4, 2, 2, 1),       # exact landing in both modes
        (3, 0.5, 2, 1),     # ICM acceleration landing exactly at boundary
        (3, 1, 0.25, 1),    # ICM deceleration landing exactly at boundary
        (1, 1, 2, 1),       # one ICM step
        (1, 0.25, 0.5, 1),  # one ACM step, zero ICM steps
    ]

    def test_boundary_contract(self):
        for generator in (ramp_steps, interval_change_ramp_steps):
            for args in self.cases:
                with self.subTest(mode=generator.__name__, args=args):
                    duration = args[0]
                    result = generator(*args)
                    positions = result["boundary_positions"]
                    intervals = result["intervals"]
                    self.assertEqual(result["step_count"], len(result["positions"]))
                    self.assertEqual(len(positions), result["step_count"] + 1)
                    np.testing.assert_array_equal(positions[:-1], result["positions"])
                    self.assertEqual(positions[-1], duration)
                    self.assertEqual(np.count_nonzero(positions == duration), 1)
                    self.assertTrue(np.all(np.diff(positions) > 0))
                    np.testing.assert_array_equal(intervals, np.diff(positions))
                    if result["step_count"]:
                        self.assertEqual(result["positions"][0], 0)
                        self.assertAlmostEqual(sum(intervals), duration)
                        self.assertEqual(result["closing_interval"], duration - result["positions"][-1])
                    else:
                        np.testing.assert_array_equal(positions, [duration])
                        self.assertEqual(len(intervals), 0)
                        self.assertIsNone(result["closing_interval"])

    def test_roundoff_boundary_is_not_duplicated(self):
        for terminal in (2.0, np.nextafter(2.0, 0.0), np.nextafter(2.0, np.inf)):
            result = boundary_step_data([0, 1, terminal], 2.0)
            np.testing.assert_array_equal(result["boundary_positions"], [0, 1, 2])
            self.assertTrue(np.all(result["intervals"] > 0))
        # A real short remainder stays visible.
        result = boundary_step_data([0, 2 - 1e-10], 2.0)
        self.assertEqual(len(result["intervals"]), 2)
        self.assertGreater(result["closing_interval"], 0)

    def test_known_generated_positions_are_preserved(self):
        # Analytic area crossings and hand-calculated iterative steps. The
        # boundary must not rescale these or lock the closing interval to 1/R_end.
        for generator, args, expected in (
            (ramp_steps, (1, 1, 3, 1), [0, (np.sqrt(5) - 1) / 2]),
            (ramp_steps, (1, 3, 1, 1), [0, (3 - np.sqrt(5)) / 2]),
            (interval_change_ramp_steps, (3, 0.5, 2, 1), [0, 2]),
            (interval_change_ramp_steps, (3, 1, 0.25, 1), [0, 1]),
        ):
            result = generator(*args)
            self.assertEqual(result["step_count"], len(expected))
            np.testing.assert_allclose(result["positions"], expected, rtol=0, atol=1e-14)
            self.assertNotAlmostEqual(result["closing_interval"], 1 / args[2])

    def test_charts_and_audio(self):
        helpers = app_helpers()
        for generator, chart in (
            (ramp_steps, "make_behavior_figure"),
            (interval_change_ramp_steps, "make_interval_change_behavior_figure"),
        ):
            for args in self.cases:
                with self.subTest(mode=generator.__name__, args=args):
                    duration = args[0]
                    result = generator(*args)
                    fig = helpers[chart](*args, result)
                    fig.canvas.draw()
                    steps, intervals = fig.axes[2:]
                    self.assertLess(steps.get_xlim()[0], 0)
                    self.assertGreater(steps.get_xlim()[1], duration)
                    np.testing.assert_array_equal(steps.lines[0].get_xdata(), [duration])
                    self.assertIn("Boundary reference", steps.lines[0].get_label())
                    if len(result["intervals"]):
                        np.testing.assert_array_equal(intervals.lines[0].get_ydata(), result["intervals"])
                        self.assertIn("Closing interval", intervals.lines[-1].get_label())
                    plt.close(fig)

                    # At a slow tempo the boundary hit is isolated. Verify the
                    # entire PCM buffer against exactly one hit per position.
                    sample_rate, tempo, gain = 8000, 40, 0.75
                    wav = helpers["render_cowbell_ramp_wav"](
                        result["boundary_positions"], duration, tempo, sample_rate, gain,
                        kick_gain=0,
                    )
                    with wave.open(io.BytesIO(wav), "rb") as source:
                        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
                    expected = np.zeros(len(pcm))
                    hit = helpers["make_cowbell_hit"](sample_rate)
                    for beat in result["boundary_positions"]:
                        start = round(beat * 60 / tempo * sample_rate)
                        expected[start:start + len(hit)] += hit * gain
                    expected_wav = helpers["audio_to_wav_bytes"](expected, sample_rate)
                    self.assertEqual(wav, expected_wav)
                    boundary_sample = round(duration * 60 / tempo * sample_rate)
                    self.assertGreaterEqual(len(pcm) - boundary_sample, len(hit))
                    self.assertTrue(np.any(pcm[boundary_sample:boundary_sample + len(hit)]))


if __name__ == "__main__":
    unittest.main()
