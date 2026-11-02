import io
import unittest
import wave

import numpy as np

from test_boundary_interval import app_helpers
from ramp.math import interval_change_ramp_steps, ramp_steps


class KickPlaybackTests(unittest.TestCase):
    def test_kicks_on_each_beat_and_one_complete_boundary_hit(self):
        helpers = app_helpers()
        sample_rate = 8000
        kick = helpers["make_kick_hit"](sample_rate)
        self.assertTrue(np.all(np.isfinite(kick)))
        self.assertGreater(np.max(abs(kick)), 0)
        self.assertEqual(kick[-1], 0)
        for tempo in (40, 120, 240):
            for generator, duration, start, end in (
                (ramp_steps, 4, 1, 4),
                (interval_change_ramp_steps, 4, 4, 1),
                (interval_change_ramp_steps, 1, 0.25, 0.5),  # no ramp steps
            ):
                with self.subTest(tempo=tempo, generator=generator.__name__, duration=duration):
                    positions = generator(duration, start, end, 1, skew=80)["boundary_positions"]
                    beat_seconds = 60 / tempo
                    # Independent reference mix includes D+1 beat kicks.
                    expected = np.zeros(int((duration * beat_seconds + 0.35) * sample_rate))
                    for beat in range(duration + 1):
                        offset = round(beat * beat_seconds * sample_rate)
                        expected[offset:offset + len(kick)] += kick * 0.65
                    cowbell = helpers["make_cowbell_hit"](sample_rate)
                    for position in positions:
                        offset = round(position * beat_seconds * sample_rate)
                        expected[offset:offset + len(cowbell)] += cowbell * 0.75
                    actual = helpers["render_cowbell_ramp_wav"](positions, duration, tempo, sample_rate)
                    self.assertEqual(actual, helpers["audio_to_wav_bytes"](expected, sample_rate))
                    with wave.open(io.BytesIO(actual)) as source:
                        boundary_offset = round(duration * beat_seconds * sample_rate)
                        self.assertGreaterEqual(source.getnframes() - boundary_offset, len(kick))


if __name__ == "__main__":
    unittest.main()
