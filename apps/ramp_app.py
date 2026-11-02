from pathlib import Path
import sys
import io
import wave

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit.components.v2 import component

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


SLIDER_DEFAULTS = {
    "duration": 4,
    "skew": 0,
    "shape_exponent": 1.0,
    "interval_shape_exponent": 1.0,
    "tempo_bpm": 88,
    "cowbell_gain": 0.75,
}

DEFAULTS = {
    **SLIDER_DEFAULTS,
    "start_rate_text": "3:1",
    "end_rate_text": "4:1",
}


for key, value in DEFAULTS.items():
    # Reassign saved values so Streamlit keeps them even when a mode hides
    # their widget. Otherwise widget cleanup deletes the inactive exponent.
    st.session_state[key] = st.session_state.get(key, value)

# Normalize saved fractional durations before creating the integer widget.
st.session_state["duration"] = max(
    1, min(16, int(float(st.session_state["duration"]) + 0.5))
)

# Equal slider distances multiply the exponent by the same factor (16 steps
# per octave). Keep saved linear-slider values and their reciprocals so existing
# sessions retain their settings and 1 stays at the midpoint.
for options_key, exponent_key in (
    ("p_slider_options", "shape_exponent"),
    ("q_slider_options", "interval_shape_exponent"),
):
    if options_key not in st.session_state:
        saved_exponent = st.session_state[exponent_key]
        st.session_state[options_key] = sorted(
            {2.0 ** (index / 16) for index in range(-64, 65)}
            | {saved_exponent, 1.0 / saved_exponent}
        )


def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def reset_double_clicked_slider():
    key = st.session_state["slider_double_click"].get("reset")
    if key in SLIDER_DEFAULTS:
        st.session_state[key] = SLIDER_DEFAULTS[key]


slider_double_click = component(
    "ramp_slider_double_click",
    js=(Path(__file__).with_name("slider_reset.js")).read_text(),
)
slider_double_click(
    key="slider_double_click",
    data=list(SLIDER_DEFAULTS),
    on_reset_change=reset_double_clicked_slider,
    height=0,
)


def draw_boundary_rhythm(steps_ax, intervals_ax, duration, result):
    """Use the same boundary-inclusive rhythm in both mode charts."""
    positions = result["boundary_positions"]
    intervals = result["intervals"]
    boundary_label = "Boundary reference (following section)"
    steps_ax.eventplot(positions[:-1], lineoffsets=1, linelengths=0.7)
    steps_ax.plot(
        [positions[-1]], [1], marker="D", color="tab:orange",
        linestyle="none", label=boundary_label,
    )
    padding = duration * 0.05
    steps_ax.set_xlim(-padding, duration + padding)
    steps_ax.set_ylim(0.4, 1.9)
    steps_ax.set_yticks([])
    steps_ax.set_title(f"Step positions — {result['step_count']} generated steps + reference")
    steps_ax.set_xlabel("Beat position t (beats from ramp start)")
    steps_ax.legend(loc="upper left", fontsize="small")
    steps_ax.grid(True)

    if len(intervals):
        numbers = np.arange(1, len(intervals) + 1)
        intervals_ax.plot(numbers, intervals, marker="o")
        intervals_ax.plot(
            [numbers[-1]], [intervals[-1]], marker="D", color="tab:orange",
            linestyle="none", label="Closing interval to boundary",
        )
        intervals_ax.legend(fontsize="small")
    else:
        intervals_ax.text(
            0.5, 0.5, "No intervals (no generated steps)",
            ha="center", va="center", transform=intervals_ax.transAxes,
        )
    intervals_ax.set_title("Intervals through boundary reference")
    intervals_ax.set_xlabel("Interval number")
    intervals_ax.set_ylabel("Beats until next step")
    intervals_ax.grid(True)


def draw_curve_steps(ax, positions, values, duration):
    """Mark the same generated steps and boundary on every time-based curve."""
    if len(positions) > 1:
        ax.scatter(
            positions[:-1], values[:-1], s=24, color="tab:blue",
            label="Final step times on target", zorder=3,
        )
    ax.scatter(
        positions[-1:], values[-1:], marker="D", color="tab:orange",
        label="Boundary reference", zorder=4,
    )
    ax.set_xlim(-0.05 * duration, 1.05 * duration)
    ax.legend(fontsize="small")


def make_behavior_figure(
    duration,
    start_rate,
    end_rate,
    shape_exponent,
    result,
    skew=0.0,
):
    t = np.linspace(0, duration, 1000)

    rate_values = rate_at(
        t,
        duration,
        start_rate,
        end_rate,
        shape_exponent,
        skew,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
    )

    rate_ax = axes[0, 0]
    interval_ax = axes[0, 1]
    steps_ax = axes[1, 0]
    intervals_ax = axes[1, 1]

    # Top-left: instantaneous rate
    rate_ax.plot(t, rate_values)
    positions = result["boundary_positions"]
    step_rates = rate_at(
        positions, duration, start_rate, end_rate, shape_exponent, skew
    )
    draw_curve_steps(rate_ax, positions, step_rates, duration)
    rate_ax.set_title("Requested rate R(t) — ACM")
    rate_ax.set_xlabel("Beat")
    rate_ax.set_ylabel("Steps per beat")
    rate_ax.grid(True)

    # Top-right: equivalent continuous interval (not the generated intervals).
    interval_ax.plot(t, 1.0 / rate_values)
    draw_curve_steps(interval_ax, positions, 1.0 / step_rates, duration)
    interval_ax.set_title("Requested interval 1 / R(t)")
    interval_ax.set_xlabel("Beat")
    interval_ax.set_ylabel("Beats per step")
    interval_ax.grid(True)

    draw_boundary_rhythm(steps_ax, intervals_ax, duration, result)

    fig.tight_layout()
    return fig


def make_area_placement_figure(
    duration, start_rate, end_rate, shape_exponent, result, skew=0.0,
):
    """ACM's integer-area crossings, kept separate from the shared overview."""
    t = np.linspace(0, duration, 1000)
    positions = result["boundary_positions"]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, accumulated_area(t, duration, start_rate, end_rate, shape_exponent, skew))
    step_areas = accumulated_area(
        positions, duration, start_rate, end_rate, shape_exponent, skew
    )
    draw_curve_steps(ax, positions, step_areas, duration)
    for k in range(result["original_step_count"]):
        ax.axhline(k, linewidth=0.5, alpha=0.25)
    ax.set_title("Requested accumulated area A(t)")
    ax.set_xlabel("Beat")
    ax.set_ylabel("Accumulated steps")
    ax.grid(True)
    fig.tight_layout()
    return fig


def make_interval_change_behavior_figure(
    duration,
    start_rate,
    end_rate,
    interval_shape_exponent,
    result,
    skew=0.0,
):
    t = np.linspace(0, duration, 1000)

    interval_values = interval_change_interval_at(
        t=t,
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        interval_shape_exponent=interval_shape_exponent,
        skew=skew,
    )

    equivalent_rate_values = interval_change_equivalent_rate_at(
        t=t,
        duration=duration,
        start_rate=start_rate,
        end_rate=end_rate,
        interval_shape_exponent=interval_shape_exponent,
        skew=skew,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
    )

    rate_ax = axes[0, 0]
    interval_ax = axes[0, 1]
    steps_ax = axes[1, 0]
    intervals_ax = axes[1, 1]

    positions = result["boundary_positions"]
    step_intervals = interval_change_interval_at(
        positions, duration, start_rate, end_rate, interval_shape_exponent, skew
    )

    interval_ax.plot(t, interval_values)
    draw_curve_steps(interval_ax, positions, step_intervals, duration)
    interval_ax.set_title("Requested interval I(t) — ICM")
    interval_ax.set_xlabel("Beat")
    interval_ax.set_ylabel("Beats per step")
    interval_ax.grid(True)

    rate_ax.plot(t, equivalent_rate_values)
    draw_curve_steps(rate_ax, positions, 1.0 / step_intervals, duration)
    rate_ax.set_title("Requested rate 1 / I(t)")
    rate_ax.set_xlabel("Beat")
    rate_ax.set_ylabel("Steps per beat")
    rate_ax.grid(True)

    draw_boundary_rhythm(steps_ax, intervals_ax, duration, result)

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


def make_kick_hit(sample_rate=44_100, duration_seconds=0.28):
    """A low, decaying pitch sweep to make the beat pulse distinct from cowbell."""
    t = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    phase = 2 * np.pi * (48 * t + (110 / 35) * (1 - np.exp(-35 * t)))
    attack = np.minimum(t / 0.002, 1.0)
    hit = np.sin(phase) * attack * np.exp(-18 * t)
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
    kick_gain=0.65,
):
    """
    Render cowbell steps with a kick on every beat and the ending boundary.

    positions are in beats.
    tempo_bpm converts beat positions to seconds.
    """
    if tempo_bpm <= 0:
        tempo_bpm = 88.0

    beat_seconds = 60.0 / tempo_bpm

    tail_seconds = 0.35
    total_seconds = duration_beats * beat_seconds + tail_seconds
    total_samples = int(total_seconds * sample_rate)

    audio = np.zeros(total_samples, dtype=np.float64)
    hit = make_cowbell_hit(sample_rate=sample_rate)

    for beat_position in positions:
        start_sample = int(round(beat_position * beat_seconds * sample_rate))

        if start_sample >= total_samples:
            continue

        usable = min(len(hit), total_samples - start_sample)
        audio[start_sample : start_sample + usable] += hit[:usable] * hit_gain

    if kick_gain:
        kick = make_kick_hit(sample_rate=sample_rate)
        # The ramp uses integer beat durations. Append the boundary separately
        # so it gets one kick, including for fractional-duration API callers.
        kick_positions = np.append(np.arange(0.0, duration_beats, 1.0), duration_beats)
        for beat_position in kick_positions:
            start_sample = int(round(beat_position * beat_seconds * sample_rate))
            usable = min(len(kick), total_samples - start_sample)
            audio[start_sample : start_sample + usable] += kick[:usable] * kick_gain

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
    st.caption("Double-click any slider to restore its initial value.")

    duration = st.slider(
        "Duration, in beats",
        min_value=1,
        max_value=16,
        step=1,
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
        shape_exponent = st.select_slider(
            "Shape exponent p",
            options=st.session_state["p_slider_options"],
            format_func=lambda p: f"{p:.4g}",
            key="shape_exponent",
            help=(
                "Area Change Mode uses g(x)=x^p. "
                "p=1 is linear. p>1 changes late. p<1 changes early. "
                "The slider is logarithmic: 1 is the midpoint, and equal distances "
                "multiply or divide p by the same factor."
            ),
        )

        interval_shape_exponent = None

    else:
        interval_shape_exponent = st.select_slider(
            "Interval shape exponent q",
            options=st.session_state["q_slider_options"],
            format_func=lambda q: f"{q:.4g}",
            key="interval_shape_exponent",
            help=(
                "Interval Change Mode uses f(x)=x^q. "
                "q=1 is linear interval change. q>1 changes late. q<1 changes early. "
                "The slider is logarithmic: 1 is the midpoint, and equal distances "
                "multiply or divide q by the same factor."
            ),
        )

        shape_exponent = None

    st.caption("Logarithmic scale · 1 is the midpoint.")

    skew = st.slider(
        "SKEW",
        min_value=-100,
        max_value=100,
        step=1,
        key="skew",
        help="Negative moves the transition earlier; positive moves it later.",
    )
    st.caption(
        "This exploratory prototype allows SKEW at exponent 1. "
        "SCRITCH disables SKEW at CURVE=0; that restriction is deliberately relaxed here."
    )

    tempo_bpm = st.slider(
        "Preview tempo, BPM",
        min_value=40,
        max_value=240,
        step=1,
        key="tempo_bpm",
    )

    cowbell_gain = st.slider(
        "Cowbell level",
        min_value=0.1,
        max_value=1.0,
        step=0.05,
        key="cowbell_gain",
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
            skew=skew,
        )
    else:
        result = interval_change_ramp_steps(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            interval_shape_exponent=interval_shape_exponent,
            skew=skew,
        )

    positions = result["boundary_positions"]
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
    st.caption(
        "Kick on every beat, including the ending boundary. "
        f"Includes one boundary reference hit at t = {duration:g} beats "
        f"(start of ramp-relative beat {duration + 1:g}, in the following section). "
        "The reference is excluded from the generated-step count."
    )

    st.subheader("Derived")
    status = result["closure_status"]
    factor = result["scale_factor"]
    factor_label = f"{factor:.9g}" if factor is not None else (
        "Not applicable" if status == "no_steps" else "Undefined"
    )
    st.metric("Closure scaling factor k", factor_label)
    if status == "scaled":
        st.info(
            f"Scaling applied: k = {factor:.9g}. Removed the leftover closing gap "
            "and stretched every complete original interval proportionally to fill DURATION. "
            "The last original step is now the boundary reference."
        )
    elif status == "natural":
        st.info("Natural boundary landing: scaling not applied (k=1). All original complete intervals are preserved.")
    elif status == "single_origin_fallback":
        st.info("Single-origin fallback: no complete interval exists to scale. Use origin + boundary with one interval of DURATION; k is undefined.")
    else:
        st.info("Zero generated steps: no scaling factor or fabricated origin. Only the boundary reference is shown and played.")
    st.caption(
        f"Original generated steps: {result['original_step_count']}; final ramp steps: "
        f"{result['step_count']} + one boundary reference. "
        "Count can change abruptly when a natural boundary landing becomes partial; "
        "no smoothing heuristic is applied."
    )

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
                    "requested start steps/beat",
                    "requested end steps/beat",
                    "shape exponent",
                    "requested accumulated step area",
                    "final ramp steps (boundary excluded)",
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
                    "requested start steps/beat",
                    "requested end steps/beat",
                    "interval shape exponent",
                    "requested start interval (beats/step)",
                    "requested end interval (beats/step)",
                    "final ramp steps (boundary excluded)",
                ],
            }
        )

    st.table({
        "Interval": [
            "Requested start: 1 / START RATE",
            "Requested end: 1 / END RATE",
            "Actual first generated interval",
            "Actual last interval (closing to boundary)",
        ],
        "Beats": [
            f"{1 / start_rate:.6f}",
            f"{1 / end_rate:.6f}",
            f"{intervals[0]:.6f}" if len(intervals) else "— (no generated steps)",
            f"{intervals[-1]:.6f}" if len(intervals) else "— (no generated steps)",
        ],
    })
    st.caption(
        "Requested continuous curves describe the generator before closure scaling. "
        "Actual timing after scaling need not match that curve or its endpoint rates."
    )


with math_col:
    st.subheader("Math")
    st.markdown("**SKEW time warp (both modes)**")
    st.latex(r"S=\operatorname{clip}(\mathrm{SKEW},-100,100)")
    st.latex(r"m=0.5+0.45\frac{S}{100},\quad x=\operatorname{clip}(t/D,0,1)")
    st.latex(r"w(x)=\frac{x}{x+\frac{m}{1-m}(1-x)}")
    st.caption(f"Current SKEW = {skew:+d}; m = {0.5 + 0.45 * skew / 100:.4g}. SKEW=0 gives w(x)=x.")

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
                "Raise warped time to p. At SKEW=0, `p = 1` is linear; larger p changes late, smaller p changes early."
            )

        with equation_col:
            st.latex(r"""
                g(x) = w(x)^p
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
                g(x) = w(x)^{{{shape_exponent:.4g}}}
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
                w\left(\frac{{t}}{{{duration:.4g}}}\right)^{{{shape_exponent:.4g}}}
                """)

        if skew == 0:
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
            st.info(
                "Nonzero SKEW uses numerical accumulation of the warped rate curve "
                "and inversion at integer area crossings. The power-curve closed form "
                "applies only at SKEW=0."
            )

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
                "Raise warped time to q. At SKEW=0, `q = 1` is linear, "
                "`q > 1` changes late, `q < 1` changes early."
            )

        with equation_col:
            st.latex(r"""
                f(x) = w(x)^q
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
                w\left(\frac{t}{D}\right)^q
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
                w\left(\frac{{t}}{{{duration:.4g}}}\right)^{{{interval_shape_exponent:.4g}}}
                """)

        st.info(
            "Interval Change Mode curves step intervals directly. The exposed value q is the actual mathematical exponent."
        )

    st.markdown("**Uniform closure scaling — after generation**")
    st.latex(r"L=t_{\mathrm{last\ original}},\quad k=D/L,\quad t'_j=k\,t_j")
    st.caption(
        "For a genuine leftover gap, discard the appended boundary and its partial interval. "
        "Stretch only the complete inter-step intervals, then promote the last original step "
        "to the boundary reference at D. Interval ratios and ordering are preserved."
    )
    st.caption(
        "Natural boundary landings keep their complete final interval with k=1. "
        "A single origin uses an explicit origin-to-boundary fallback; zero-step ICM "
        "does not acquire an origin. Requested endpoint rates are not locked."
    )

with charts_col:
    st.subheader("ACM Behavior" if mode == "Area Change Mode" else "ICM Behavior")

    if mode == "Area Change Mode":
        behavior_fig = make_behavior_figure(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            shape_exponent=shape_exponent,
            skew=skew,
            result=result,
        )
    else:
        behavior_fig = make_interval_change_behavior_figure(
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            interval_shape_exponent=interval_shape_exponent,
            skew=skew,
            result=result,
        )

    st.caption(
        "Top: requested continuous targets, with final step times sampled on those curves. "
        "These samples do not represent realized rates after scaling. "
        "Bottom: actual final positions and intervals used by listings and playback."
    )
    st.pyplot(behavior_fig, clear_figure=True)
    plt.close(behavior_fig)

    if mode == "Area Change Mode":
        with st.expander("ACM placement details"):
            st.caption(
                "The original generator places steps at integer area crossings before closure "
                "scaling. Blue markers show final step times sampled on the requested area "
                "curve and may no longer hit integer crossings. The orange reference marks "
                "the following section."
            )
            area_fig = make_area_placement_figure(
                duration, start_rate, end_rate, shape_exponent, result, skew
            )
            st.pyplot(area_fig, clear_figure=True)
            plt.close(area_fig)

    details_left, details_right = st.columns(2, gap="medium")

    with details_left:
        with st.expander("Step positions"):
            for index, position in enumerate(positions, start=1):
                if index == len(positions):
                    st.write(f"Boundary reference (following section): {position:.6f} beats")
                else:
                    st.write(f"{index:>2}: {position:.6f} beats")

    with details_right:
        with st.expander("Intervals"):
            for index, interval in enumerate(intervals, start=1):
                if index == len(intervals):
                    st.write(f"{index:>2} → boundary: {interval:.6f} beats (closing interval)")
                else:
                    st.write(f"{index:>2} → {index + 1}: {interval:.6f} beats")
            if not len(intervals):
                st.caption("No intervals: no generated ramp steps.")
