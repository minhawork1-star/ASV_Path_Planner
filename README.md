# Adaptive ASV Positioning Planner for Acoustic-Aided AUV Localisation (IMU-only)

Simulation of a single surface vehicle (**ASV**) that supports two underwater vehicles (**AUVs**) on a
lawnmower survey. Each AUV carries **only an IMU** (3-axis gyroscope + accelerometer) — no compass, DVL, or
pressure sensor — so its dead-reckoned position drifts without bound. The ASV keeps a communication-free
"shadow" of each AUV's covariance, decides **when and which AUV to ping**, positions itself with a
sampling-based **Model-Predictive Path-Integral (MPPI)** planner, and delivers an acoustic
**USBL 3-D position fix** that collapses the drift. No machine learning is used.

## Result (this case)

| Metric | AUV 0 | AUV 1 |
|---|---|---|
| True cross-track error (mean) | 1.98 m | 1.86 m |
| Fixes delivered / lost | 320 | 319 |
| Max ASV–AUV slant range (< 500 m) | 172 m | 186 m |
| Covariance consistency (ANEES) | 1.01 | 0.81 |

Fleet: **639 USBL fixes delivered, 0 lost**, full survey completed in ~838 s. Figures in
[`Result/vertical_mppi/`](Result/vertical_mppi).

## Run

```bash
pip install -r requirements.txt
python mission_corridor_vertical.py
```

Running the script bare reproduces the result above; outputs (plots + trajectory data) are written to a
`Result/` folder next to the script. The behaviour is controlled by environment variables (see the comments
at the top of `mission_corridor_vertical.py`); the baked defaults are the configuration reported here.

## Contents

- `mission_corridor_vertical.py` — the simulation (AUV EKF navigation, USBL acoustic model, ASV MPPI
  planner, covariance-triggered ping scheduler).
- `mss_env.py` — the marine-vehicle environment (Otter surface craft + REMUS-class AUV dynamics, sensors).
- `reports/` — the project report (`report1_asv_planner.tex`) with its figures and bibliography.
- `Result/vertical_mppi/` — result figures for the reported run.

## Method (brief)

- **AUV navigation:** a 15-state EKF dead-reckons on the IMU, aided only by IMU-intrinsic corrections
  (gravity levelling for roll/pitch, course-over-ground for yaw). Guidance is estimate-fed (steers on the
  vehicle's own estimate, never ground truth).
- **ASV planner:** a per-AUV shadow covariance is propagated from public knowledge only; a ping is triggered
  when a shadow's covariance trace crosses a threshold; the MPPI cost trades the next fix's information gain
  (D-optimality) against keeping the whole fleet within USBL range.
- **Acoustic link:** the USBL measures range (round-trip time) and bearing (array phase); the delayed 3-D
  fix is fused as an out-of-sequence measurement.

See `reports/report1_asv_planner.tex` for the full description, derivations, results and analysis.
