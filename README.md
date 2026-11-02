# Ramp prototype

Run from the project root:

```sh
uv sync
uv run streamlit run apps/ramp_app.py --server.runOnSave true
```

Requires Python 3.13 or newer and uv. Open <http://localhost:8501>.
Stop the server with Ctrl+C.

## Current state — November 1, 2026

This Streamlit prototype compares two ramp generators:

- **Area Change Mode (ACM):** integrate a continuous rate curve and place steps
  at integer area crossings, using exponent p.
- **Interval Change Mode (ICM):** walk forward by the current continuous step
  interval, using exponent q. Infeasible inputs can generate zero ramp steps.

Both use transition-shaping SKEW, followed by the proportional closure described
below. The current implementation does not use the shelved count-preserving
positional-SKEW experiment.

| Control | Default | Behavior |
|---|---|---|
| Duration | 4 beats | Whole beats, 1–16; saved fractional values round to the nearest integer, with halves up |
| Start / end rates | `3:1` / `4:1` | Ratios such as `3:2`, in steps per beat |
| Exponents p and q | 1 | Logarithmic sliders from 1/16 to 16, with 1 at the midpoint |
| SKEW | 0 | Shared slider from −100 to +100 |
| Preview tempo | **88 BPM** | 40–240 BPM |
| Cowbell level | 0.75 | Relative cowbell level in the kick/cowbell mix |

Double-click any slider to restore its default. Reset restores all defaults,
including tempo and cowbell level. The independent p/q values survive mode
switching and unrelated reruns.

Both modes keep the same graph layout:

| | Left | Right |
|---|---|---|
| Top | Requested continuous rate | Requested continuous interval |
| Bottom | Final step positions | Actual final intervals through the boundary |

The heading identifies ACM or ICM Behavior. All time-based curves include final
step-time markers and a distinct orange boundary reference. ACM's accumulated
area graph is under **ACM placement details**. Position and interval listings
use the same final sequence as the graphs and playback.

Playback mixes a synthetic cowbell at each final step with a kick on every beat,
including beat positions 0 and DURATION. The ending reference is played once,
with enough audio tail for both instruments.

## Transition-shaping SKEW

```text
S = clamp(SKEW, -100, 100)
m = 0.5 + 0.45 * S / 100
x = clamp(t / DURATION, 0, 1)
w(x) = x / (x + (m / (1 - m)) * (1 - x))
progress = w(x) ** exponent
```

ACM interpolates start to end rate using this progress; ICM interpolates their
reciprocal intervals. Zero SKEW preserves the unwarped curve, and continuous
endpoint values stay fixed. Negative SKEW moves the transition earlier; positive
SKEW moves it later. For deceleration, an earlier slowdown can move generated
steps later even though the transition happens earlier. SKEW can change the
original generated count.

For exploration, SKEW remains active at exponent 1; this deliberately differs
from SCRITCH's rule disabling it at CURVE=0. ACM keeps its analytic integral at
zero SKEW. Nonzero SKEW uses adaptive Gaussian accumulation with cached panels
and local inversion, avoiding integration from the origin for every step. ICM
retains its interval-walk algorithm.

## Proportional closure

ACM and ICM keep their existing transition-shaping exponent/SKEW algorithms.
After generation, a shared closure transform removes only a genuine partial
closing interval:

- ACM naturally lands at the boundary when its accumulated cycle total is a
  positive integer within `1e-9` cycles.
- ICM naturally lands when its proposed next position equals DURATION within
  `1e-9` beats. Detection happens in the original interval walk.
- A natural landing preserves all complete intervals, including the legitimate
  interval to the boundary, with scale factor `k=1`. A shorter final interval
  alone is not evidence of truncation.
- For a remainder, discard the inspection/playback boundary and its partial
  closing interval. Retain the intervals between original generated positions.
  With `L = last_original_position > 0`, multiply them all by `k=DURATION/L`.
  The last original step becomes the boundary reference exactly at DURATION.
- An origin without any complete interval uses an explicit origin + boundary
  fallback with one interval of DURATION. Its scale factor is undefined.
- Zero-generated-step ICM remains empty: show/play the boundary reference but
  do not invent an origin or interval.

DURATION is in beats. The boundary is the start of ramp-relative beat
DURATION+1 and belongs to the following section. It is shown/played once and
excluded from the final ramp-step count. In a scaled remainder case, promoting
an original step to the boundary reduces that count by one. Counts can change
abruptly when a natural landing becomes partial; no smoothing heuristic hides
that behavior.

Uniform scaling preserves retained interval ratios and ordering. It does not
lock endpoint intervals or guarantee requested endpoint rates or exact
agreement with the requested continuous curves. The top charts show those
requested targets with final step times sampled on them, not realized rate or
interval estimates. Bottom charts, listings, and audio all use the final rhythm.
The app displays closure status, factor, and original versus final step counts.

Generator result fields `positions`, `boundary_positions`, `intervals`,
`closing_interval`, and `step_count` describe the final rhythm. Inspection fields
`original_positions`, `original_step_count`, and `original_intervals` describe
original steps strictly before the boundary. `retained_intervals` additionally
includes the naturally fitting final interval in exact cases; for remainders it
excludes the discarded partial interval. `natural_boundary_landing`,
`closure_status`, `scaling_applied`, and `scale_factor` describe the transform.

Compare these settings at SKEW=0 and the default tempo of 88 BPM:

| Mode | Duration | Start → end | Exact fit | Scaled remainder |
|---|---|---|---|---|
| ACM | 4 | `1:1` → `3:1` | p=1, k=1 | p=2, k≈1.061272583 |
| ICM | 3 | `1:2` → `2:1` | q=1, intervals [2, 1] | q=2, k=1.5, interval [3] |

Also try SKEW=-80/+80, exponents 0.5/2, and reversing the rates. The existing
logarithmic exponent controls, double-click reset, cowbell steps, and kick on
each beat through the boundary remain available.

## Changes made on November 1, 2026

- Exposed the full closing interval and following-section boundary reference in
  inspection and cowbell playback, keeping the reference out of the ramp count.
- Fixed exponent state loss on mode switches, limited Duration to integer beats,
  and made both exponent controls logarithmic.
- Added transition-shaping SKEW to both generators, curves, equations, and audio,
  with numerical ACM integration for nonzero SKEW.
- Added per-slider double-click reset and a kick pulse through the boundary.
- Overlaid steps on the continuous graphs, aligned the four quadrants across
  modes, moved accumulated area into placement details, and added mode headings.
- Explored and reverted a count-preserving positional interpretation of SKEW.
  The accepted version retains transition-shaping SKEW.
- Added uniform proportional closure, explicit exact-fit/fallback/empty statuses,
  final count reporting, target-versus-actual labels, and regression coverage.
- Changed the default and reset tempo from 120 to **88 BPM**.

## Validation

Run regression checks:

```sh
MPLCONFIGDIR=/tmp/ramp-mpl uv run python -m unittest discover -s tests
node --test tests/test_slider_reset.mjs
```

Latest local validation: **22 Python tests and 1 JavaScript event test pass**.
Coverage includes natural and near-natural boundary landings, scaled remainders,
interval ratios, acceleration/deceleration, SKEW and exponent extremes, origin
fallbacks, zero-step ICM, graph/listing agreement, audio mixing and boundary tails,
and slider state/reset behavior. In 240 comparisons against the pre-closure
snapshot, the original transition-shaped generation matched exactly; natural-fit
final sequences were unchanged. Rendered charts were visually inspected.

Double-click handling has automated frontend and Streamlit callback tests; a
live browser gesture was not verified. No browser connection is needed to run
the checks above.
