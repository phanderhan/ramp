"""Focused SKEW regression checks; run with unittest discover -s tests."""
import unittest
from unittest.mock import patch

import numpy as np

from test_boundary_interval import app_helpers
from ramp.math import (
    _numerical_accumulation, accumulated_area, interval_change_interval_at,
    interval_change_ramp_steps, interval_change_equivalent_rate_at,
    interval_shape_progress, ramp_steps, rate_at,
    shaped_progress, skew_time,
)


class SkewTests(unittest.TestCase):
    exponents = (1 / 16, 0.5, 1, 2, 16)

    def test_zero_skew_keeps_analytic_path_and_existing_results(self):
        t = np.linspace(0, 4, 33)
        with patch("ramp.math._numerical_accumulation", side_effect=AssertionError("zero skew must be analytic")):
            for exponent in self.exponents:
                np.testing.assert_allclose(
                    rate_at(t, 4, 1, 3, exponent, skew=0),
                    1 + 2 * (t / 4)**exponent, rtol=0, atol=1e-15,
                )
                np.testing.assert_allclose(
                    accumulated_area(t, 4, 1, 3, exponent, skew=0),
                    t + 8 * (t / 4)**(exponent + 1) / (exponent + 1),
                    rtol=0, atol=1e-14,
                )
                np.testing.assert_allclose(
                    interval_change_interval_at(t, 4, 1, 3, exponent, skew=0),
                    1 + (1 / 3 - 1) * (t / 4)**exponent, rtol=0, atol=1e-15,
                )
            np.testing.assert_allclose(ramp_steps(1, 1, 3, 1, skew=0)["positions"], [0, (np.sqrt(5) - 1) / 2])
            np.testing.assert_array_equal(interval_change_ramp_steps(3, 1, 0.25, 1, skew=0)["positions"], [0, 1])

    def test_warp_endpoints_midpoint_clamping_and_progress(self):
        x = np.linspace(0, 1, 1001)
        for skew in (-100, -50, 0, 50, 100):
            m = 0.5 + 0.45 * skew / 100
            self.assertEqual(skew_time(0, skew), 0)
            self.assertEqual(skew_time(1, skew), 1)
            self.assertAlmostEqual(float(skew_time(m, skew)), 0.5)
            np.testing.assert_array_equal(skew_time([-1, 2], skew), [0, 1])
            for exponent in self.exponents:
                for progress in (shaped_progress, interval_shape_progress):
                    values = progress(x, exponent, skew)
                    self.assertTrue(np.all(np.isfinite(values)))
                    self.assertTrue(np.all(np.diff(values) >= 0))
                    self.assertEqual(values[0], 0)
                    self.assertEqual(values[-1], 1)
        np.testing.assert_array_equal(skew_time(x, -200), skew_time(x, -100))
        np.testing.assert_array_equal(skew_time(x, 200), skew_time(x, 100))

    def test_transition_direction_and_continuous_endpoints(self):
        x = np.linspace(0.01, 0.99, 99)
        for start, end in ((1, 4), (4, 1)):
            for exponent in self.exponents:
                for curve, first, last in (
                    (rate_at, start, end),
                    (interval_change_interval_at, 1 / start, 1 / end),
                ):
                    baseline = (curve(x * 8, 8, start, end, exponent) - first) / (last - first)
                    early = (curve(x * 8, 8, start, end, exponent, -100) - first) / (last - first)
                    late = (curve(x * 8, 8, start, end, exponent, 100) - first) / (last - first)
                    self.assertTrue(np.all(early >= baseline - 1e-14))
                    self.assertTrue(np.any(early > baseline + 1e-8))
                    self.assertTrue(np.all(late <= baseline + 1e-14))
                    self.assertTrue(np.any(late < baseline - 1e-8))
                    for skew in (-100, 100):
                        np.testing.assert_allclose(curve([0, 8], 8, start, end, exponent, skew), [first, last], atol=1e-14)
                for generator in (ramp_steps, interval_change_ramp_steps):
                    early = generator(8, start, end, exponent, -100)["original_positions"]
                    late = generator(8, start, end, exponent, 100)["original_positions"]
                    count = min(len(early), len(late))
                    difference = early[:count] - late[:count]
                    self.assertTrue(np.all(difference <= 1e-12) if end > start else np.all(difference >= -1e-12))
                    self.assertTrue(np.any(abs(difference) > 1e-8))

    def test_numerical_accumulation_against_analytic_zero_skew(self):
        t = np.linspace(0, 16, 257)
        for start, end in ((1 / 32, 32), (32, 1 / 32)):
            for exponent in self.exponents:
                numeric = _numerical_accumulation(16, start, end, exponent, 0)
                analytic = accumulated_area(t, 16, start, end, exponent)
                np.testing.assert_allclose(numeric.area_at(t), analytic, rtol=1e-11, atol=2e-11)
                targets = np.arange(int(analytic[-1]))
                positions = numeric.times_for_areas(targets)
                np.testing.assert_allclose(accumulated_area(positions, 16, start, end, exponent), targets, atol=2e-10)

    def test_nonzero_skew_accuracy_and_tighter_stability(self):
        t = np.linspace(0, 16, 257)
        for start, end in ((1 / 32, 32), (32, 1 / 32)):
            for exponent in self.exponents:
                for skew in (-100, 100):
                    with self.subTest(start=start, exponent=exponent, skew=skew):
                        normal = _numerical_accumulation(16, start, end, exponent, skew)
                        tight = _numerical_accumulation(16, start, end, exponent, skew, rtol=1e-14)
                        np.testing.assert_allclose(normal.area_at(t), tight.area_at(t), atol=2e-10, rtol=1e-12)
                        targets = np.arange(int(normal.areas[-1]))
                        positions = normal.times_for_areas(targets)
                        np.testing.assert_allclose(positions, tight.times_for_areas(targets), atol=2e-10, rtol=1e-12)
                        np.testing.assert_allclose(tight.area_at(positions), targets, atol=2e-10)
                        self.assertIs(normal, _numerical_accumulation(16, start, end, exponent, skew))
                        if exponent == 1:
                            # Independent exact integral of w(x)=x/(a+(1-a)x).
                            a = (0.5 + 0.45 * skew / 100) / (0.5 - 0.45 * skew / 100)
                            z = t / 16
                            integral = z / (1 - a) - a / (1 - a)**2 * np.log1p((1 - a) * z / a)
                            exact = 16 * (start * z + (end - start) * integral)
                            np.testing.assert_allclose(normal.area_at(t), exact, atol=2e-10, rtol=1e-11)

    def test_icm_walk_and_zero_step_behavior(self):
        for skew in (-100, 0, 100):
            for generator in (ramp_steps, interval_change_ramp_steps):
                exact = generator(4, 2, 2, 1, skew)
                self.assertEqual(exact["step_count"], 8)
                np.testing.assert_allclose(exact["boundary_positions"], np.arange(9) / 2, atol=1e-13)
                self.assertTrue(np.all(exact["intervals"] > 0))
                single = generator(1, 1, 1, 1, skew)
                self.assertEqual(single["step_count"], 1)
                np.testing.assert_array_equal(single["boundary_positions"], [0, 1])
                np.testing.assert_array_equal(single["intervals"], [1])
            result = interval_change_ramp_steps(8, 1, 4, 0.5, skew)
            positions = result["original_positions"]
            np.testing.assert_allclose(np.diff(positions), interval_change_interval_at(positions[:-1], 8, 1, 4, 0.5, skew), atol=1e-14)
            empty = interval_change_ramp_steps(1, 0.25, 0.5, 1, skew)
            self.assertEqual(empty["step_count"], 0)
            np.testing.assert_array_equal(empty["boundary_positions"], [1])
            self.assertEqual(len(empty["intervals"]), 0)

    def test_skewed_charts_boundary_and_playback(self):
        helpers = app_helpers()
        for generator, chart, curve in (
            (ramp_steps, "make_behavior_figure", rate_at),
            (interval_change_ramp_steps, "make_interval_change_behavior_figure", interval_change_equivalent_rate_at),
        ):
            for skew in (-100, 100):
                for start, end in ((1, 4), (4, 1)):
                    result = generator(4, start, end, 1, skew)
                    positions = result["boundary_positions"]
                    self.assertEqual(positions[-1], 4)
                    self.assertEqual(np.count_nonzero(positions == 4), 1)
                    self.assertTrue(np.all(np.diff(positions) > 0))
                    self.assertAlmostEqual(sum(result["intervals"]), 4)
                    self.assertEqual(result["step_count"], len(positions) - 1)
                    fig = helpers[chart](4, start, end, 1, result, skew)
                    x, y = fig.axes[0].lines[0].get_data()
                    np.testing.assert_allclose(y, curve(x, 4, start, end, 1, skew))
                    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), 1 / y)
                    self.assertIn("rate", fig.axes[0].get_title())
                    self.assertIn("interval", fig.axes[1].get_title())
                    self.assertGreater(fig.axes[2].get_xlim()[1], 4)
                    if generator is ramp_steps:
                        area_fig = helpers["make_area_placement_figure"](4, start, end, 1, result, skew)
                        np.testing.assert_allclose(area_fig.axes[0].lines[0].get_ydata(), accumulated_area(x, 4, start, end, 1, skew))
                        markers = np.concatenate([c.get_offsets() for c in area_fig.axes[0].collections])
                        np.testing.assert_array_equal(markers[:, 0], positions)
                        np.testing.assert_allclose(markers[:, 1], accumulated_area(positions, 4, start, end, 1, skew))
                        helpers["plt"].close(area_fig)
                    helpers["plt"].close(fig)
                    # Compare rendering with an independently assembled hit buffer.
                    sample_rate = 8000
                    audio = np.zeros(int((4 * 0.5 + 0.35) * sample_rate))
                    hit = helpers["make_cowbell_hit"](sample_rate)
                    for position in positions:
                        offset = round(position * 0.5 * sample_rate)
                        audio[offset:offset + len(hit)] += hit * 0.75
                    self.assertEqual(
                        helpers["render_cowbell_ramp_wav"](positions, 4, 120, sample_rate, kick_gain=0),
                        helpers["audio_to_wav_bytes"](audio, sample_rate),
                    )


if __name__ == "__main__":
    unittest.main()
