from pathlib import Path
import sys
import io
import wave

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ramp.math import (
    accumulated_area,
    interval_change_equivalent_rate_at,
    interval_change_interval_at,
    interval_change_ramp_steps,
    parse_rate,
    ramp_steps,
    rate_at,
)

st.set_page_config(
    page_title="SCRITCH RAMP Math",
    layout="wide",
)


DEFAULTS = {
    "duration": 4.0,
    "start_rate_text": "3:1",
    "end_rate_text": "4:1",
    "shape_exponent": 1.0,
    "interval_shape_exponent": 1.0,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def make_behavior_figure(
    duration,
    start_rate,
    end_rate,
    shape_exponent,
    result,
):
    t = np.linspace(0, duration, 1000)

    rate_values = rate_at(
        t,
        duration,
        start_rate,
        end_rate,
        shape_exponent,
    )

    area_values = accumulated_area(
        t,
        duration,
        start_rate,
        end_rate,
        shape_exponent,
    )

    positions = result["positions"]
    intervals = result["intervals"]
    step_count = result["step_count"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
    )

    rate_ax = axes[0, 0]
    area_ax = axes[0, 1]
    steps_ax = axes[1, 0]
    intervals_ax = axes[1, 1]

    # Top-left: instantaneous rate
    rate_ax.plot(t, rate_values)
    rate_ax.set_title("Instantaneous rate R(t)")
    rate_ax.set_xlabel("Beat")
    rate_ax.set_ylabel("Steps per beat")
    rate_ax.grid(True)

    # Top-right: accumulated area
    area_ax.plot(t, area_values)

    for k in range(step_count):
        area_ax.axhline(k, linewidth=0.5, alpha=0.25)

    area_ax.set_title("Accumulated area A(t)")
    area_ax.set_xlabel("Beat")
    area_ax.set_ylabel("Accumulated steps")
    area_ax.grid(True)

    # Bottom-left: generated step positions
    steps_ax.eventplot(
        positions,
        lineoffsets=1,
        linelengths=0.7,
    )
    steps_ax.set_xlim(0, duration)
    steps_ax.set_yticks([])
    steps_ax.set_title(f"Generated step positions — {step_count} steps")
    steps_ax.set_xlabel("Beat")
    steps_ax.grid(True)

    # Bottom-right: intervals
    if len(intervals) > 0:
        interval_numbers = np.arange(1, len(intervals) + 1)
        intervals_ax.plot(interval_numbers, intervals, marker="o")
    else:
        intervals_ax.text(
            0.5,
            0.5,
            "No intervals",
            ha="center",
            va="center",
            transform=intervals_ax.transAxes,
        )

    intervals_ax.set_title("Inter-step intervals")
    intervals_ax.set_xlabel("Interval number")
    intervals_ax.set_ylabel("Beats until next step")
    intervals_ax.grid(True)

    fig.tight_layout()
    return fig


def make_interval_change_behavior_figure(
    duration,
    start_rate,
    end_rate,
    interval_shape_exponent,
    result,
):
    t = np.linspace(0, duration, 1000)

    interval_values = interval_change_interval_at(
        t=t,
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        interval_shape_exponent=interval_shape_exponent,
    )

    equivalent_rate_values = interval_change_equivalent_rate_at(
        t=t,
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        interval_shape_exponent=interval_shape_exponent,
    )

    positions = result["positions"]
    intervals = result["intervals"]
    step_count = result["step_count"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
    )

    interval_ax = axes[0, 0]
    rate_ax = axes[0, 1]
    steps_ax = axes[1, 0]
    intervals_ax = axes[1, 1]

    interval_ax.plot(t, interval_values)
    interval_ax.set_title("Step interval I(t)")
    interval_ax.set_xlabel("Beat")
    interval_ax.set_ylabel("Beats per step")
    interval_ax.grid(True)

    rate_ax.plot(t, equivalent_rate_values)
    rate_ax.set_title("Equivalent rate 1 / I(t)")
    rate_ax.set_xlabel("Beat")
    rate_ax.set_ylabel("Steps per beat")
    rate_ax.grid(True)

    steps_ax.eventplot(
        positions,
        lineoffsets=1,
        linelengths=0.7,
    )
    steps_ax.set_xlim(0, duration)
    steps_ax.set_yticks([])
    steps_ax.set_title(f"Generated step positions — {step_count} steps")
    steps_ax.set_xlabel("Beat")
    steps_ax.grid(True)

    if len(intervals) > 0:
        interval_numbers = np.arange(1, len(intervals) + 1)
        intervals_ax.plot(interval_numbers, intervals, marker="o")
    else:
        intervals_ax.text(
            0.5,
            0.5,
            "No intervals",
            ha="center",
            va="center",
            transform=intervals_ax.transAxes,
        )

    intervals_ax.set_title("Inter-step intervals")
    intervals_ax.set_xlabel("Interval number")
    intervals_ax.set_ylabel("Beats until next step")
    intervals_ax.grid(True)

    fig.tight_layout()
    return fig


def make_cowbell_hit(sample_rate=44_100, duration_seconds=0.14):
    """
    Make a short synthetic cowbell-ish hit.

    This is not trying to be a beautiful instrument.
    It is trying to make step timing obvious.
    """
    sample_count = int(sample_rate * duration_seconds)
    t = np.arange(sample_count) / sample_rate

    # Metallic-ish partials.
    osc_1 = np.sign(np.sin(2 * np.pi * 540 * t))
    osc_2 = np.sign(np.sin(2 * np.pi * 800 * t))
    osc_3 = np.sin(2 * np.pi * 1200 * t)

    body = 0.45 * osc_1 + 0.35 * osc_2 + 0.20 * osc_3

    # Sharp attack, fast decay.
    envelope = np.exp(-t * 32)

    hit = body * envelope

    # Tiny noise tick for attack definition.
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, 0.05, sample_count) * np.exp(-t * 90)

    hit = hit + noise

    # Avoid clicks at the end of the sample.
    fade_samples = min(256, len(hit))
    hit[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    return hit


def audio_to_wav_bytes(audio, sample_rate=44_100):
    """
    Convert a floating-point mono audio buffer in roughly -1..1 to WAV bytes.
    """
    audio = np.asarray(audio, dtype=np.float64)

    max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_abs > 0:
        audio = audio / max_abs

    audio_i16 = np.asarray(audio * 32767, dtype=np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_i16.tobytes())

    buffer.seek(0)
    return buffer.read()


def render_cowbell_ramp_wav(
    positions,
    duration_beats,
    tempo_bpm,
    sample_rate=44_100,
    hit_gain=0.75,
):
    """
    Render generated ramp step positions as cowbell hits.

    positions are in beats.
    tempo_bpm converts beat positions to seconds.
    """
    if tempo_bpm <= 0:
        tempo_bpm = 120.0

    beat_seconds = 60.0 / tempo_bpm

    tail_seconds = 0.35
    total_seconds = duration_beats * beat_seconds + tail_seconds
    total_samples = int(total_seconds * sample_rate)

    audio = np.zeros(total_samples, dtype=np.float64)
    hit = make_cowbell_hit(sample_rate=sample_rate)

    for beat_position in positions:
        start_sample = int(round(beat_position * beat_seconds * sample_rate))
        end_sample = start_sample + len(hit)

        if start_sample >= total_samples:
            continue

        usable = min(len(hit), total_samples - start_sample)
        audio[start_sample : start_sample + usable] += hit[:usable] * hit_gain

    return audio_to_wav_bytes(audio, sample_rate=sample_rate)


st.title("SCRITCH RAMP Math Proof")

mode = st.radio(
    "Mode",
    [
        "Area Change Mode",
        "Interval Change Mode",
    ],
    horizontal=True,
)

if mode == "Area Change Mode":
    st.markdown("""
        This mockup treats **RAMP** as a continuous **rate curve**.

        In **Area Change Mode**, the demo exposes the mathematical shape parameter
        directly as **shape exponent p**. The ramp curves instantaneous rate,
        accumulates area under that rate curve, and places steps at integer
        area crossings. Later, SCRITCH's user-facing **CURVE** control can be mapped onto this parameter.
        """)
else:
    st.markdown("""
        This mockup treats **RAMP** as a continuous **interval curve**.

        In **Interval Change Mode**, the demo exposes the mathematical shape parameter
        directly as **interval exponent q**. The ramp curves the time interval
        between steps, then places each next step by adding the current interval. Later, SCRITCH's user-facing **CURVE** control can be mapped onto this parameter.
        """)

params_col, math_col, charts_col = st.columns(
    [1.2, 2.0, 3.6],
    gap="large",
)


with params_col:
    st.subheader("Parameters")

    st.button("Reset", on_click=reset_defaults)

    duration = st.slider(
        "Duration, in beats",
        min_value=1.0,
        max_value=16.0,
        step=0.25,
        key="duration",
    )

    start_rate_text = st.text_input(
        "Start rate",
        help='Use Scritch-style ratios like "4:1", "3:2", "1:32", or "32:1".',
        key="start_rate_text",
    )

    end_rate_text = st.text_input(
        "End rate",
        help='Use Scritch-style ratios like "4:1", "3:2", "1:32", or "32:1".',
        key="end_rate_text",
    )

    if mode == "Area Change Mode":
        shape_exponent = st.slider(
            "Shape exponent p",
            min_value=0.0625,
            max_value=16.0,
            step=0.0625,
            key="shape_exponent",
            help=(
                "Area Change Mode uses g(x)=x^p. "
                "p=1 is linear. p>1 changes late. p<1 changes early."
            ),
        )

        interval_shape_exponent = None

    else:
        interval_shape_exponent = st.slider(
            "Interval shape exponent q",
            min_value=0.0625,
            max_value=16.0,
            step=0.0625,
            key="interval_shape_exponent",
            help=(
                "Interval Change Mode uses f(x)=x^q. "
                "q=1 is linear interval change. q>1 changes late. q<1 changes early."
            ),
        )

        shape_exponent = None

    tempo_bpm = st.slider(
        "Preview tempo, BPM",
        min_value=40,
        max_value=240,
        value=120,
        step=1,
    )

    cowbell_gain = st.slider(
        "Cowbell level",
        min_value=0.1,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )

    try:
        start_rate = parse_rate(start_rate_text)
        end_rate = parse_rate(end_rate_text)
    except ValueError as error:
        st.error(f"Invalid rate: {error}")
        st.stop()

    if mode == "Area Change Mode":
        result = ramp_steps(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            shape_exponent=shape_exponent,
        )
    else:
        result = interval_change_ramp_steps(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            interval_shape_exponent=interval_shape_exponent,
        )

    positions = result["positions"]
    intervals = result["intervals"]

    st.divider()

    st.subheader("Audio")

    wav_bytes = render_cowbell_ramp_wav(
        positions=positions,
        duration_beats=duration,
        tempo_bpm=tempo_bpm,
        hit_gain=cowbell_gain,
    )

    st.audio(wav_bytes, format="audio/wav")

    st.subheader("Derived")

    if mode == "Area Change Mode":
        st.table(
            {
                "Value": [
                    f"{start_rate:.6g}",
                    f"{end_rate:.6g}",
                    f"{shape_exponent:.4g}",
                    f"{result['total_area']:.6f}",
                    f"{result['step_count']}",
                ],
                "Meaning": [
                    "start steps/beat",
                    "end steps/beat",
                    "shape exponent",
                    "accumulated step area",
                    "generated steps",
                ],
            }
        )
    else:
        st.table(
            {
                "Value": [
                    f"{start_rate:.6g}",
                    f"{end_rate:.6g}",
                    f"{interval_shape_exponent:.4g}",
                    f"{result['start_interval']:.6f}",
                    f"{result['end_interval']:.6f}",
                    f"{result['step_count']}",
                ],
                "Meaning": [
                    "start steps/beat",
                    "end steps/beat",
                    "interval shape exponent",
                    "start beats/step",
                    "end beats/step",
                    "generated steps",
                ],
            }
        )


with math_col:
    st.subheader("Math")

    if mode == "Area Change Mode":
        st.markdown("#### Area Change Mode")

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Normalize time**")
            st.caption("Convert beat position `t` into a 0–1 position inside the ramp.")

        with equation_col:
            st.latex(r"""
                x = \frac{t}{D}
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Shape progress**")
            st.caption(
                "Bend normalized time. `p = 1` is linear, `p > 1` changes late, `p < 1` changes early."
            )

        with equation_col:
            st.latex(r"""
                g(x) = x^p
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Rate curve**")
            st.caption(
                "Interpolate from the start rate to the end rate using shaped progress."
            )

        with equation_col:
            st.latex(r"""
                R(t) =
                R_{\mathrm{start}}
                +
                \left(
                R_{\mathrm{end}} - R_{\mathrm{start}}
                \right)
                g\left(\frac{t}{D}\right)
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Accumulated area**")
            st.caption(
                "Count how much step-density has accumulated from the start to time `t`."
            )

        with equation_col:
            st.latex(r"""
                A(t) = \int_0^t R(u)\,du
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Step placement**")
            st.caption(
                "Place a step whenever accumulated area crosses the next integer."
            )

        with equation_col:
            st.latex(r"""
                A(t) = k,\quad k = 0,1,2,\ldots
                """)

        st.divider()

        st.markdown("#### Current model")

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Current shape**")
            st.caption("This is the shape function generated by the current `p` value.")

        with equation_col:
            st.latex(rf"""
                g(x) = x^{{{shape_exponent:.4g}}}
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Current rate curve**")
            st.caption(
                "This is the concrete rate curve using the current duration and parsed rates."
            )

        with equation_col:
            st.latex(rf"""
                R(t) =
                {start_rate:.4g}
                +
                \left(
                {end_rate:.4g} - {start_rate:.4g}
                \right)
                \left(
                \frac{{t}}{{{duration:.4g}}}
                \right)^{{{shape_exponent:.4g}}}
                """)

        with st.expander("Closed-form accumulated area"):
            st.caption(
                "Because Area Change Mode uses a power curve, accumulated area has a closed form."
            )

            st.latex(rf"""
                A(t) =
                {start_rate:.4g}t
                +
                \left(
                {end_rate:.4g} - {start_rate:.4g}
                \right)
                {duration:.4g}
                \frac{{
                \left(
                \frac{{t}}{{{duration:.4g}}}
                \right)^{{{shape_exponent + 1:.4g}}}
                }}{{{shape_exponent + 1:.4g}}}
                """)

    else:
        st.markdown("#### Interval Change Mode")

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Normalize time**")
            st.caption("Convert beat position `t` into a 0–1 position inside the ramp.")

        with equation_col:
            st.latex(r"""
                x = \frac{t}{D}
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Shape interval progress**")
            st.caption(
                "Bend normalized time. `q = 1` is linear, "
                "`q > 1` changes late, `q < 1` changes early."
            )

        with equation_col:
            st.latex(r"""
                f(x) = x^q
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Step interval curve**")
            st.caption("This mode curves the interval between steps, not the rate.")

        with equation_col:
            st.latex(r"""
                I(t) =
                \frac{1}{R_{\mathrm{start}}}
                +
                \left(
                \frac{1}{R_{\mathrm{end}}}
                -
                \frac{1}{R_{\mathrm{start}}}
                \right)
                \left(\frac{t}{D}\right)^q
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Iterative placement**")
            st.caption(
                "Each step is placed by adding the current local interval to the current time."
            )

        with equation_col:
            st.latex(r"""
                t_0 = 0
                \qquad
                t_{n+1} = t_n + I(t_n)
                """)

        st.divider()

        st.markdown("#### Current model")

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Current interval endpoints**")
            st.caption("The start and end rates become start and end intervals.")

        with equation_col:
            st.latex(rf"""
                I_{{\mathrm{{start}}}} = \frac{{1}}{{{start_rate:.4g}}}
                \qquad
                I_{{\mathrm{{end}}}} = \frac{{1}}{{{end_rate:.4g}}}
                """)

        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Current interval shape**")
            st.caption("This is the actual exponent `q` used by Interval Change Mode.")

        with equation_col:
            st.latex(rf"""
                q = {interval_shape_exponent:.4g}
                """)
        label_col, equation_col = st.columns([1.0, 1.35], gap="medium")

        with label_col:
            st.markdown("**Current interval curve**")
            st.caption("This is the concrete interval curve for the selected settings.")

        with equation_col:
            st.latex(rf"""
                I(t) =
                {1 / start_rate:.4g}
                +
                \left(
                {1 / end_rate:.4g}
                -
                {1 / start_rate:.4g}
                \right)
                \left(\frac{{t}}{{{duration:.4g}}}\right)^{{{interval_shape_exponent:.4g}}}
                """)

        st.info(
            "Interval Change Mode curves step intervals directly. The exposed value q is the actual mathematical exponent."
        )

with charts_col:
    st.subheader("Behavior")

    if mode == "Area Change Mode":
        behavior_fig = make_behavior_figure(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            shape_exponent=shape_exponent,
            result=result,
        )
    else:
        behavior_fig = make_interval_change_behavior_figure(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            interval_shape_exponent=interval_shape_exponent,
            result=result,
        )

    st.pyplot(behavior_fig, clear_figure=True)
    plt.close(behavior_fig)

    details_left, details_right = st.columns(2, gap="medium")

    with details_left:
        with st.expander("Step positions"):
            for index, position in enumerate(positions, start=1):
                st.write(f"{index:>2}: {position:.6f}")

    with details_right:
        with st.expander("Intervals"):
            for index, interval in enumerate(intervals, start=1):
                st.write(f"{index:>2} → {index + 1}: {interval:.6f}")
