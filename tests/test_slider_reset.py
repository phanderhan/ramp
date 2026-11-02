import json
from pathlib import Path
import unittest

from streamlit.components.v2.bidi_component.main import _make_trigger_id
from streamlit.testing.v1 import AppTest


class SliderResetTests(unittest.TestCase):
    def trigger_reset(self, app, key):
        # AppTest doesn't yet expose v2 component interaction helpers. Send the
        # same component state and trigger protobufs as the browser frontend.
        widget_states = app._tree.get_widget_states()
        component = app.get("bidi_component")[0]
        widget_states.widgets.add(id=component.proto.id, json_value="{}")
        widget_states.widgets.add(
            id=_make_trigger_id(component.proto.id, "events"),
            json_trigger_value=json.dumps([{"event": "reset", "value": key}]),
        )
        app._run(widget_states)
        self.assertFalse(app.exception)

    def test_each_slider_resets_independently_including_repeated_events(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "apps/ramp_app.py")
        ).run()
        defaults = {
            "duration": 4, "skew": 0, "shape_exponent": 1.0,
            "interval_shape_exponent": 1.0, "tempo_bpm": 88, "cowbell_gain": 0.75,
        }
        for key, changed in (
            ("duration", 7), ("skew", 35), ("shape_exponent", 4.0),
            ("tempo_bpm", 180), ("cowbell_gain", 0.35),
            ("interval_shape_exponent", 2.0),
            ("interval_shape_exponent", 0.5),
        ):
            with self.subTest(key=key, changed=changed):
                if key == "interval_shape_exponent":
                    app.radio[0].set_value("Interval Change Mode").run()
                if key in ("shape_exponent", "interval_shape_exponent"):
                    widget = app.select_slider(key=key)
                else:
                    widget = app.slider(key=key)
                widget.set_value(changed).run()
                before = {name: app.session_state[name] for name in defaults}
                self.trigger_reset(app, key)
                self.assertEqual(app.session_state[key], defaults[key])
                for other in defaults:
                    if other != key:
                        self.assertEqual(app.session_state[other], before[other])
                # The trigger must not reset a later ordinary edit.
                app.slider(key="tempo_bpm").set_value(150).run()
                self.assertEqual(app.slider(key="tempo_bpm").value, 150)

        before_rate = app.text_input[0].value
        self.trigger_reset(app, "start_rate_text")
        self.assertEqual(app.text_input[0].value, before_rate)
        app.button[0].click().run()
        self.assertEqual(app.slider(key="tempo_bpm").value, 88)
        self.assertEqual(app.slider(key="cowbell_gain").value, 0.75)


if __name__ == "__main__":
    unittest.main()
