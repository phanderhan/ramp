from pathlib import Path
import math
import unittest

from streamlit.testing.v1 import AppTest


class ModeStateTests(unittest.TestCase):
    def test_exponent_sliders_are_logarithmic_and_preserve_legacy_values(self):
        app_path = str(Path(__file__).resolve().parents[1] / "apps/ramp_app.py")
        for mode, key, options_key in (
            ("Area Change Mode", "shape_exponent", "p_slider_options"),
            ("Interval Change Mode", "interval_shape_exponent", "q_slider_options"),
        ):
            with self.subTest(mode=mode):
                app = AppTest.from_file(app_path).run()
                app.radio[0].set_value(mode).run()
                slider = app.select_slider(key=key)
                values = app.session_state[options_key]
                self.assertEqual(values[0], 1 / 16)
                self.assertEqual(values[-1], 16)
                self.assertEqual(values[len(values) // 2], 1)
                for left, right in zip(values, values[1:]):
                    self.assertAlmostEqual(math.log2(right / left), 1 / 16)
                slider.set_value(0.5).run()
                self.assertEqual(app.session_state[key], 0.5)
                self.assertFalse(app.exception)
                self.assertTrue(any("Logarithmic scale" in c.value for c in app.caption))

                legacy = AppTest.from_file(app_path)
                legacy.session_state[key] = 2.5
                legacy.run()
                legacy.radio[0].set_value(mode).run()
                self.assertEqual(legacy.select_slider(key=key).value, 2.5)
                values = legacy.session_state[options_key]
                self.assertEqual(values[len(values) // 2], 1)
                self.assertFalse(legacy.exception)

    def test_duration_clamps_saved_values_to_whole_beats(self):
        for saved, expected in ((4.25, 4), (4.5, 5), (0.25, 1), (16.75, 16)):
            with self.subTest(saved=saved):
                app = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "apps/ramp_app.py")
                )
                app.session_state["duration"] = saved
                app.run()
                self.assertFalse(app.exception)
                duration = app.slider(key="duration")
                self.assertEqual(duration.value, expected)
                self.assertIsInstance(duration.value, int)
                self.assertEqual(duration.proto.step, 1)
                app.radio[0].set_value("Interval Change Mode").run()
                self.assertEqual(app.slider(key="duration").value, expected)

    def test_exponents_survive_mode_switches_and_reset(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "apps/ramp_app.py")
        ).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.select_slider(key="shape_exponent").value, 1.0)
        app.radio[0].set_value("Interval Change Mode").run()
        self.assertEqual(app.select_slider(key="interval_shape_exponent").value, 1.0)

        app.select_slider(key="interval_shape_exponent").set_value(2.0).run()
        app.radio[0].set_value("Area Change Mode").run()
        app.select_slider(key="shape_exponent").set_value(4.0).run()
        # An unrelated rerun while q is hidden must also preserve its value.
        app.slider(key="duration").set_value(5).run()
        app.radio[0].set_value("Interval Change Mode").run()
        self.assertEqual(app.select_slider(key="interval_shape_exponent").value, 2.0)
        app.radio[0].set_value("Area Change Mode").run()
        self.assertEqual(app.select_slider(key="shape_exponent").value, 4.0)

        app.button[0].click().run()
        self.assertEqual(app.select_slider(key="shape_exponent").value, 1.0)
        app.radio[0].set_value("Interval Change Mode").run()
        self.assertEqual(app.select_slider(key="interval_shape_exponent").value, 1.0)
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
