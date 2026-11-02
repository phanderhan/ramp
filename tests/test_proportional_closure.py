"""Uniform closure after the original transition-shaped ACM/ICM generation."""
from pathlib import Path
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from test_boundary_interval import app_helpers
from ramp.math import (
    accumulated_area, interval_change_interval_at,
    interval_change_ramp_steps, ramp_steps,
)


class ProportionalClosureTests(unittest.TestCase):
    def assert_closure(self, result, duration):
        raw = result["original_positions"]
        final = result["boundary_positions"]
        intervals = result["intervals"]
        self.assertEqual(final[-1], duration)
        self.assertEqual(np.count_nonzero(final == duration), 1)
        self.assertTrue(np.all(np.diff(final) > 0))
        self.assertTrue(np.all(intervals > 0))
        np.testing.assert_array_equal(intervals, np.diff(final))
        np.testing.assert_array_equal(result["positions"], final[:-1])
        self.assertEqual(result["step_count"], len(final) - 1)
        self.assertEqual(result["original_step_count"], len(raw))
        if len(raw):
            self.assertEqual(final[0], 0)
            self.assertAlmostEqual(sum(intervals), duration)
        else:
            np.testing.assert_array_equal(final, [duration])
            self.assertEqual(result["step_count"], 0)
        status = result["closure_status"]
        if status == "scaled":
            factor = duration / raw[-1]
            self.assertTrue(result["scaling_applied"])
            self.assertFalse(result["natural_boundary_landing"])
            self.assertEqual(result["scale_factor"], factor)
            self.assertEqual(result["step_count"], len(raw) - 1)
            self.assertEqual(len(intervals), len(raw) - 1)
            np.testing.assert_allclose(final[:-1], raw[:-1] * factor, atol=1e-14)
            np.testing.assert_allclose(intervals / np.diff(raw), factor, rtol=2e-11)
            np.testing.assert_array_equal(result["retained_intervals"], np.diff(raw))
            if len(intervals) > 1:
                np.testing.assert_allclose(intervals[1:] / intervals[:-1], np.diff(raw)[1:] / np.diff(raw)[:-1], rtol=2e-11)
        elif status == "natural":
            self.assertFalse(result["scaling_applied"])
            self.assertEqual(result["scale_factor"], 1)
            self.assertTrue(result["natural_boundary_landing"])
            np.testing.assert_array_equal(result["positions"], raw)
            np.testing.assert_array_equal(final, np.append(raw, duration))
            np.testing.assert_array_equal(intervals, result["retained_intervals"])
        else:
            self.assertIsNone(result["scale_factor"])
            self.assertFalse(result["scaling_applied"])

    def test_exact_and_near_exact_landings_stay_unchanged(self):
        for generator in (ramp_steps, interval_change_ramp_steps):
            for skew in (-100, 0, 100):
                for exponent in (.5, 1, 2):
                    result = generator(4, 2, 2, exponent, skew)
                    self.assertEqual(result["closure_status"], "natural")
                    np.testing.assert_allclose(result["boundary_positions"], np.arange(9) / 2, atol=1e-13)
                    self.assert_closure(result, 4)
        for delta in (-5e-10, 5e-10):
            # ACM tolerance is in accumulated cycles.
            result = ramp_steps(4, (8 + delta) / 4, (8 + delta) / 4, 1)
            self.assertEqual(result["closure_status"], "natural")
            self.assert_closure(result, 4)
            # ICM tolerance is in beats of the proposed next position.
            rate = 8 / (4 + delta)
            result = interval_change_ramp_steps(4, rate, rate, 1)
            self.assertEqual(result["closure_status"], "natural")
            self.assert_closure(result, 4)
        # Nonconstant, nonzero-SKEW curves can also land naturally. Construct
        # a mean rate of two using ACM's linear dependence on endpoint rates.
        for exponent in (.5, 2):
            for skew in (-60, 60):
                weight = float(accumulated_area(1, 1, 1, 2, exponent, skew)) - 1
                for start, end in ((1, 1 + 1 / weight), (1 + 1 / (1 - weight), 1)):
                    result = ramp_steps(4, start, end, exponent, skew)
                    self.assertEqual(result["closure_status"], "natural")
                    self.assert_closure(result, 4)
        # Acceleration can have a legitimate shorter last interval.
        for generator, args, expected in (
            (ramp_steps, (1, 1, 3, 1), [0, (np.sqrt(5) - 1) / 2, 1]),
            (interval_change_ramp_steps, (3, .5, 2, 1), [0, 2, 3]),
            (interval_change_ramp_steps, (3, 1, .25, 1), [0, 1, 3]),
        ):
            result = generator(*args)
            self.assertEqual(result["closure_status"], "natural")
            np.testing.assert_allclose(result["boundary_positions"], expected, atol=1e-14)

    def test_remainders_preserve_original_generation_and_interval_ratios(self):
        scaled_by_direction = {}
        for generator in (ramp_steps, interval_change_ramp_steps):
            for start, end in ((1, 3), (3, 1), (1 / 32, 32), (32, 1 / 32)):
                for exponent in (.0625, .5, 1, 2, 16):
                    for skew in (-80, 0, 80):
                        with self.subTest(mode=generator.__name__, start=start, exponent=exponent, skew=skew):
                            result = generator(4, start, end, exponent, skew)
                            self.assert_closure(result, 4)
                            raw = result["original_positions"]
                            if generator is ramp_steps:
                                np.testing.assert_allclose(accumulated_area(raw, 4, start, end, exponent, skew), np.arange(len(raw)), atol=2e-10)
                            elif len(raw):
                                np.testing.assert_allclose(np.diff(raw), interval_change_interval_at(raw[:-1], 4, start, end, exponent, skew), atol=1e-14)
                            if result["scaling_applied"]:
                                scaled_by_direction[(generator.__name__, end > start)] = True
        self.assertEqual(len(scaled_by_direction), 4)

    def test_origin_fallback_and_empty_icm(self):
        for generator in (ramp_steps, interval_change_ramp_steps):
            # ACM has no complete interval. ICM's one-origin case instead has
            # a naturally fitting first interval, which must remain exact.
            args = (1, .25, .5, 1) if generator is ramp_steps else (1, 1, .25, 1)
            result = generator(*args)
            expected = "single_origin_fallback" if generator is ramp_steps else "natural"
            self.assertEqual(result["closure_status"], expected)
            np.testing.assert_array_equal(result["boundary_positions"], [0, 1])
            np.testing.assert_array_equal(result["intervals"], [1])
            self.assert_closure(result, 1)
        for skew in (-100, 0, 100):
            result = interval_change_ramp_steps(1, .25, .5, 1, skew)
            self.assertEqual(result["closure_status"], "no_steps")
            self.assert_closure(result, 1)

    def test_count_discontinuity_is_not_smoothed(self):
        for generator, exact_rate, partial_rate in (
            (ramp_steps, 2, (8 - 1e-7) / 4),
            (interval_change_ramp_steps, 2, 8 / (4 + 1e-7)),
        ):
            exact = generator(4, exact_rate, exact_rate, 1)
            partial = generator(4, partial_rate, partial_rate, 1)
            self.assertEqual(exact["closure_status"], "natural")
            self.assertEqual(partial["closure_status"], "scaled")
            self.assertEqual(exact["step_count"], 8)
            self.assertEqual(partial["step_count"], 7)
            self.assert_closure(partial, 4)

    def test_chart_markers_and_listings_use_final_scaled_positions(self):
        helpers = app_helpers()
        for generator, chart in (
            (ramp_steps, "make_behavior_figure"),
            (interval_change_ramp_steps, "make_interval_change_behavior_figure"),
        ):
            result = generator(4, 1, 3, .5, 80)
            self.assertEqual(result["closure_status"], "scaled")
            fig = helpers[chart](4, 1, 3, .5, result, 80)
            for axis in fig.axes[:2]:
                self.assertIn("Requested", axis.get_title())
                offsets = np.concatenate([c.get_offsets() for c in axis.collections])
                np.testing.assert_array_equal(offsets[:, 0], result["boundary_positions"])
            np.testing.assert_array_equal(fig.axes[3].lines[0].get_ydata(), result["intervals"])
            helpers["plt"].close(fig)

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/ramp_app.py")).run()
        self.assertTrue(any("Natural boundary landing" in item.value for item in app.info))
        app.select_slider(key="shape_exponent").set_value(.5).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Scaling applied:" in item.value for item in app.info))
        result = ramp_steps(4, 3, 4, .5)
        self.assertEqual(app.metric[0].value, f"{result['scale_factor']:.9g}")
        text = [item.value for item in app.markdown]
        self.assertTrue(any(f"{result['positions'][1]:.6f} beats" in value for value in text))
        self.assertTrue(any("Boundary reference (following section): 4.000000 beats" in value for value in text))
        app.slider(key="duration").set_value(1)
        app.text_input[0].set_value("1:4")
        app.text_input[1].set_value("1:2").run()
        self.assertTrue(any("Single-origin fallback" in item.value for item in app.info))
        app.radio[0].set_value("Interval Change Mode").run()
        self.assertTrue(any("Zero generated steps" in item.value for item in app.info))
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
