"""
mission_corridor_final.py — FINAL corridor mission (self-contained).

This is a frozen copy of stage4_acoustic_physics.py (the acoustic nav) with the winning corridor
configuration baked in as defaults, so running it bare reproduces the RESULT in New mission plan/RUNS.md:
  2 AUVs, side-by-side blocks with a central corridor; ASV rides the corridor delivering 3-D USBL fixes.
  IMU-only AUV (raw strapdown + gravity leveling + course-over-ground heading aid + 3-D USBL) -- NO
  depth/compass/DVL. Autopilot closed on the AUV's own EKF ESTIMATE (no truth oracle). 500 m-rated USBL.

Result (both ASV modes, both AUVs): true cross-track ~2 m (p95 ~4 m), yaw ~5 deg, 0 USBL lost.

The COG heading aid (USE_COG=1) is the fix for the earlier "blind-yaw" wobble (yaw drifted to 14-52 deg
with no heading aid -> the est-fed autopilot crabbed off-lane). See RUNS.md "DIAGNOSIS".

Run (bare = MPPI ASV):
    py mission_corridor_final.py
Flip the ASV planner to the centroid baseline:
    ASV_NAIVE=1 CENTROID_HOLD=1 RUN_TAG=corridor_final_centroid py mission_corridor_final.py
Any baked default below can still be overridden by setting the same env var before launch.
Cross-track is graded against the TRUE trajectory (scoring only; never fed to control/estimation).
"""
import os, math, collections

# --- WINNING CORRIDOR CONFIG baked in as defaults (setdefault => env vars still override) ---
for _k, _v in {
    "N_AUVS": "2", "LEG_LEN": "200", "AUV_SEP": "260",   # geometry: two blocks + central corridor
    "SHALLOW_MAXR": "500",                                # 500 m-rated USBL (serves the ~460 m corridor)
    "USE_COG": "1",                                       # course-over-ground heading aid (the yaw fix)
    "AUTOPILOT_ON_EST": "1",                              # steer on the ESTIMATE (realistic, no truth oracle)
    "DROPOUT_MODEL": "none", "MAC_TIMING": "0", "RAY_PHYSICS": "0",  # clean acoustics (isolate positioning)
    "N_TICKS_MAX": "45000",                               # long enough to complete the survey
    "ASV_NAIVE": "0", "COST_BOWL": "1", "MPPI_MULTI_AUV": "sum", "COVERAGE_WEIGHT": "400",  # MPPI planner
    "RUN_TAG": "corridor_final",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# locate mss_env by walking up (location-proof; this script lives one level down in "New mission plan/")
import sys as _sys_boot
_here_boot = os.path.dirname(os.path.abspath(__file__))
_ASV_ROOT = None
while _here_boot != os.path.dirname(_here_boot):
    if os.path.exists(os.path.join(_here_boot, "mss_env.py")):
        _ASV_ROOT = _here_boot
        _sys_boot.path.insert(0, _here_boot)
        break
    _here_boot = os.path.dirname(_here_boot)
import mss_env


# ============================================================================
# CONSTANTS (original fixA5 values — tuned for 2 AUVs)
# ============================================================================
DT          = 0.02
# DEPTH_REF: AUV survey depth. Env-configurable (default 5.0 = original). Deeper AUVs make the USBL geometry
# non-benign -- the ASV must hold a horizontal standoff to avoid the overhead cone -> the Fisher-info geometry
# actually drives positioning. Flows to mss_env.make(depth_ref=...) and Z_START.
DEPTH_REF   = float(os.environ.get("DEPTH_REF", 40.0))   # DEEP by default (ASV-side-trigger study): range binds
# Depth-from-IMU: command a slow depth profile (instead of holding DEPTH_REF) so depth is
# observable and the network's strong vertical axis (dp_z) actually has something to track.
DEPTH_AMP    = 3.0        # m, peak depth excursion around DEPTH_REF
DEPTH_PERIOD = 75.0       # s, one up-down cycle (gentle, survey-like)
SPEED       = 1.0
TURN_R      = 30.0
LANE_SPACING= 60.0

# USBL noise (fixA5)
C_SOUND     = 1500.0
SIG_R       = 0.10
# SIG_A: USBL bearing accuracy [deg]. Env-tunable so the model can match a REAL device: the Blueprint SeaTrac X150
# (24-32 kHz, 1 km rated) specifies ~+/-1 deg (typ 2% of acoustic range). Default 0.5 reproduces all earlier runs.
SIG_A       = float(os.environ.get("SIG_A", "0.5"))
SIG_GPS     = 1.0
LOSS        = 0.279
MAXR        = 150.0
# SHALLOW_MAXR: operational range limit [m] of a REALISTIC CHEAP SHALLOW USBL (Blueprint SeaTrac X150 class, ~1 km).
# The free-field sonar equation OVER-predicts range in shallow water -- it ignores reverberation/multipath -- so a
# real low-cost shallow device is rated ~1 km, not the ~km-many the bare equation gives. Modeled HONESTLY as detection
# PHYSICS: a soft logistic rolloff of P(detect) centered here (the reverberation-limited envelope), using the TRUE
# slant range like all propagation does. A ping to an AUV beyond it simply gets NO REPLY -- a wasted cycle; the ASV
# never learns the true range, so nothing is cheated. inf = OFF (deep-water class = current behaviour EXACTLY).
SHALLOW_MAXR = float(os.environ.get("SHALLOW_MAXR", "inf"))
# OPER_RANGE: the ASV's operational USBL range for PLANNING/reporting -- the shallow device's rating if set, else the
# old hard MAXR. The MPPI coverage cost + range plots key off THIS (not the stale MAXR=150) so the ASV positions
# against the real reach. SHALLOW_MAXR=inf -> OPER_RANGE=MAXR=150 -> old behaviour EXACTLY.
OPER_RANGE  = SHALLOW_MAXR if math.isfinite(SHALLOW_MAXR) else MAXR
OUT_P       = 0.03
OUT_B       = 8.0
PROC        = 0.05
DOWNLINK_LEG    = True
ELEV_CONE_MIN_DEG = 5.0

# ============================================================================
# STAGE-A ACOUSTIC PROPAGATION PHYSICS (literature-backed; replaces the fake hard-cutoff+flat-dropout link).
# Every term below is a named, published model. NO timing terms are added here (Stage B). PHYSICS_OFF=1 restores
# the old hard-cutoff behaviour so results can be checked against stage3. Refs:
#   Urick 1983 (spreading/sonar eq); Francois & Garrison 1982 + Thorp 1967 (absorption); Wenz 1962 + Stojanovic
#   2007 (ambient noise fit); Mackenzie 1981 (sound speed); Van Trees (CRLB); Rayleigh fading outage (comms).
# ============================================================================
PHYSICS_OFF     = os.environ.get("PHYSICS_OFF", "0") == "1"      # 1 -> old sl>MAXR + flat LOSS (faithfulness)
PHYSICS_SELFTEST= os.environ.get("PHYSICS_SELFTEST", "0") == "1" # 1 -> plot/validate physics vs literature, exit
FREQ_KHZ   = float(os.environ.get("FREQ_KHZ",  "25.0"))   # USBL carrier (kHz); typical medium-freq USBL 18-36 kHz
SL_DB      = float(os.environ.get("SL_DB",     "190.0"))  # source level, dB re 1 uPa @ 1 m
DT_DB      = float(os.environ.get("DT_DB",     "15.0"))   # detection threshold / required SNR (dB)
DI_DB      = float(os.environ.get("DI_DB",     "20.0"))   # receiver array directivity index / processing gain (dB)
BW_HZ      = float(os.environ.get("BW_HZ",     "2000.0")) # receiver noise bandwidth (Hz); narrowband ranging
TEMP_C     = float(os.environ.get("TEMP_C",    "10.0"))   # water temperature (deg C)  [Mackenzie/F-G input]
SAL_PPT    = float(os.environ.get("SAL_PPT",   "35.0"))   # salinity (ppt)
PH_WATER   = float(os.environ.get("PH_WATER",  "8.0"))    # pH  [F-G boric-acid term]
SEA_STATE  = float(os.environ.get("SEA_STATE", "2.0"))    # 0 (calm) .. 6 (rough); -> wind speed -> Wenz noise
SPREAD_N   = float(os.environ.get("SPREAD_N",  "1.5"))    # spreading exponent: 2=spherical,1=cylindrical,1.5 mid
ABS_MODEL  = os.environ.get("ABS_MODEL", "fg")            # "fg"=Francois-Garrison(1982), "thorp"=Thorp(1967)
SNR_REF_DB = float(os.environ.get("SNR_REF_DB", "40.0"))  # SNR at which the base range/bearing sigmas hold (CRLB)


def sound_speed_mackenzie(T=TEMP_C, S=SAL_PPT, D=0.0):
    """Sound speed c [m/s]. Mackenzie (1981) 9-term eq. Valid 2-30 C, 25-40 ppt, 0-8000 m. D = depth [m]."""
    return (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3
            + 1.340*(S - 35.0) + 1.630e-2*D + 1.675e-7*D**2
            - 1.025e-2*T*(S - 35.0) - 7.139e-13*T*D**3)


def _alpha_thorp(f_khz):
    """Absorption alpha [dB/km], Thorp (1967). f in kHz. Simple f-only fallback (valid ~0.1-50 kHz)."""
    f2 = f_khz*f_khz
    return 0.11*f2/(1.0+f2) + 44.0*f2/(4100.0+f2) + 2.75e-4*f2 + 0.003


def _alpha_francois_garrison(f_khz, T=TEMP_C, S=SAL_PPT, pH=PH_WATER, depth_m=0.0):
    """Absorption alpha [dB/km], Francois & Garrison (1982): boric acid + MgSO4 + pure-water relaxations.
    f in kHz, T deg C, S ppt, depth m. Standard total-absorption formula."""
    c = 1412.0 + 3.21*T + 1.19*S + 0.0167*depth_m          # F-G internal sound-speed approx
    # Boric acid
    A1 = (8.86/c) * 10.0**(0.78*pH - 5.0)
    f1 = 2.8*math.sqrt(S/35.0) * 10.0**(4.0 - 1245.0/(T + 273.0))
    P1 = 1.0
    # Magnesium sulphate
    A2 = 21.44*(S/c)*(1.0 + 0.025*T)
    f2 = (8.17*10.0**(8.0 - 1990.0/(T + 273.0))) / (1.0 + 0.0018*(S - 35.0))
    P2 = 1.0 - 1.37e-4*depth_m + 6.2e-9*depth_m**2
    # Pure water
    if T <= 20.0:
        A3 = 4.937e-4 - 2.590e-5*T + 9.11e-7*T**2 - 1.50e-8*T**3
    else:
        A3 = 3.964e-4 - 1.146e-5*T + 1.45e-7*T**2 - 6.5e-10*T**3
    P3 = 1.0 - 3.83e-5*depth_m + 4.9e-10*depth_m**2
    f = f_khz
    return (A1*P1*f1*f*f/(f1*f1 + f*f)
            + A2*P2*f2*f*f/(f2*f2 + f*f)
            + A3*P3*f*f)


def absorption_db_per_km(f_khz, depth_m=0.0):
    if ABS_MODEL == "thorp":
        return _alpha_thorp(f_khz)
    return _alpha_francois_garrison(f_khz, TEMP_C, SAL_PPT, PH_WATER, depth_m)


def wind_speed_from_sea_state(ss):
    """Wind speed [m/s] from sea state (~Beaufort number). Beaufort empirical w = 0.836 * B^1.5."""
    return 0.836 * max(0.0, ss)**1.5


def ambient_noise_psd_db(f_khz, sea_state=SEA_STATE, shipping=0.5):
    """Ambient-noise power spectral density [dB re 1 uPa^2/Hz] vs frequency. Stojanovic (2007) 4-component fit to
    the Wenz (1962) curves: turbulence + shipping + wind/waves + thermal. f in kHz."""
    f = max(f_khz, 1e-3)
    w = wind_speed_from_sea_state(sea_state)
    lg = math.log10
    nt = 17.0 - 30.0*lg(f)                                              # turbulence (dominant f<10 Hz)
    ns = 40.0 + 20.0*(shipping - 0.5) + 26.0*lg(f) - 60.0*lg(f + 0.03)  # shipping
    nw = 50.0 + 7.5*math.sqrt(w) + 20.0*lg(f) - 40.0*lg(f + 0.4)        # wind/waves (dominant ~0.5-50 kHz)
    nth = -15.0 + 20.0*lg(f)                                            # thermal (dominant >50 kHz)
    return 10.0*lg(10.0**(nt/10) + 10.0**(ns/10) + 10.0**(nw/10) + 10.0**(nth/10))


def transmission_loss_db(R_m, f_khz=FREQ_KHZ, depth_m=0.0):
    """One-way transmission loss [dB]: geometric spreading + absorption. R in m."""
    R = max(R_m, 1.0)
    return 10.0*SPREAD_N*math.log10(R) + absorption_db_per_km(f_khz, depth_m)*(R/1000.0)


def link_snr_db(R_m, f_khz=FREQ_KHZ, sea_state=SEA_STATE, depth_m=0.0):
    """Sonar-equation SNR [dB] for one transponder link leg: SNR = SL - TL - (NL_psd + 10log10 BW) + DI."""
    tl = transmission_loss_db(R_m, f_khz, depth_m)
    nl = ambient_noise_psd_db(f_khz, sea_state) + 10.0*math.log10(BW_HZ)
    return SL_DB - tl - nl + DI_DB


def p_detect(R_m, f_khz=FREQ_KHZ, sea_state=SEA_STATE, depth_m=0.0):
    """Probability the link (both legs, symmetric) is detected. Rayleigh-fading outage:
    P = exp(-gamma_th/gamma_bar) = exp(-10^((DT - SNR)/10)). Smooth 1->0; 'max range' emerges where SNR->DT."""
    snr = link_snr_db(R_m, f_khz, sea_state, depth_m)
    return math.exp(-10.0**((DT_DB - snr)/10.0))


# ---- Residual packet-error (dropout) model: BER->PER waterfall + Rician fading + empirical floor ----
# Backed: BER formulas [Proakis, Digital Communications]; Rician multipath fading [Stojanovic]; PER floor from
# published modem field trials. DROPOUT_MODEL selects: 'per' (this), 'rayleigh' (Stage-A p_detect), 'none'.
DROPOUT_MODEL = os.environ.get("DROPOUT_MODEL", "per")     # per | rayleigh | none
MOD           = os.environ.get("MOD", "fsk")               # fsk (robust, default) | psk
NBITS         = int(os.environ.get("NBITS", "256"))        # fix-telegram size [bits]
K_FACTOR      = float(os.environ.get("K_FACTOR", "10.0"))  # Rician K (direct/multipath power); high=strong LOS
PER_FLOOR     = float(os.environ.get("PER_FLOOR", "0.02")) # residual PER even at infinite SNR (Doppler/sync/hw)
CODE_GAIN_DB  = float(os.environ.get("CODE_GAIN_DB", "0.0"))  # FEC coding gain, applied as an SNR shift [dB]

# ---- Stage B: MAC / interrogation-cycle timing + half-duplex channel (default OFF -> stage4 reproduces exactly) ----
# Realistic two-way cycle: T_PING + prop_up + TAT + T_REPLY + prop_down + PROC + [downlink T_REPLY+prop] + GUARD.
# Every term is a real physical/modem quantity (no padding). Cycle is RANGE-DEPENDENT (prop legs = R/c). The
# half-duplex channel is EVENT-DRIVEN (the ASV waits for the reply) -> no true-range prediction, no interference.
MAC_TIMING = os.environ.get("MAC_TIMING", "0") == "1"
MODEM_BPS  = float(os.environ.get("MODEM_BPS", "1000.0"))  # acoustic bit rate (real 80 bps-15 kbps); telegram=NBITS/bps
T_PING     = float(os.environ.get("T_PING", "0.02"))       # interrogation ranging-signal duration [s]
TAT        = float(os.environ.get("TAT", "0.05"))          # transponder turn-around time [s]
GUARD      = float(os.environ.get("GUARD", "0.10"))        # multipath/reverb settle before channel reuse [s]


def usbl_cycle_time(slant_m, depth_m=0.0):
    """Full two-way interrogation-cycle duration [s] at slant range R. Range-dependent (prop legs scale with R).
    T_REPLY = NBITS/MODEM_BPS (telegram from bit-rate). DOWNLINK relays the fix to the AUV (another telegram+leg)."""
    one_way = slant_m / (C_SOUND if PHYSICS_OFF else sound_speed_mackenzie(TEMP_C, SAL_PPT, depth_m))
    t_reply = NBITS / MODEM_BPS
    cyc = T_PING + one_way + TAT + t_reply + one_way + PROC
    if DOWNLINK_LEG:
        cyc += t_reply + one_way
    return cyc + GUARD


def _ber(gamma_lin):
    """Bit error rate vs per-bit SNR gamma (linear). Non-coherent BFSK or coherent BPSK [Proakis]."""
    if MOD == "psk":
        return 0.5 * math.erfc(math.sqrt(max(gamma_lin, 0.0)))          # Q(sqrt(2*gamma)) = 0.5*erfc(sqrt(gamma))
    return 0.5 * math.exp(-0.5 * max(gamma_lin, 0.0))                    # non-coherent BFSK


def _rician_gain(rng, k=None):
    """One normalized power-gain sample g (E[g]=1) for a Rician channel with factor K (default K_FACTOR; Stage C
    passes a geometry-derived K). h = sqrt(K/(K+1)) + CN(0, 1/(K+1)); g = |h|^2."""
    K = K_FACTOR if k is None else max(k, 1e-3)
    los = math.sqrt(K / (K + 1.0))
    s = math.sqrt(1.0 / (2.0 * (K + 1.0)))
    hr = los + s * rng.normal(); hi = s * rng.normal()
    return hr * hr + hi * hi


def packet_success_prob(R_m, f_khz, sea_state, depth_m, rng, k=None, snr_penalty_db=0.0):
    """P(fix packet gets through) for ONE ping. Draws a Rician fade, maps instantaneous SNR -> BER -> PER, floors
    it. gamma_mean from the sonar-equation link SNR (treated as Eb/N0, bandwidth-matched); CODE_GAIN_DB shifts it.
    Stage C: `k` overrides the Rician K (from multipath geometry) and `snr_penalty_db` subtracts a Doppler loss."""
    snr_db = link_snr_db(R_m, f_khz, sea_state, depth_m) + CODE_GAIN_DB - snr_penalty_db
    gamma_mean = 10.0 ** (snr_db / 10.0)
    gamma_inst = gamma_mean * _rician_gain(rng, k)       # per-packet Rician fade (geometry K if given)
    ber = _ber(gamma_inst)
    per = 1.0 - (1.0 - ber) ** NBITS
    per = max(per, PER_FLOOR)                            # residual floor
    return 1.0 - per


# ============================================================================
# STAGE C: propagation GEOMETRY (analytic) -- refraction (bending) + multipath (bouncing) + Doppler.
# Refs: Urick; Jensen et al. "Computational Ocean Acoustics" (linear-SSP circular rays); Brekhovskikh & Lysanov
# (image-method multipath); Stojanovic (Doppler). Default OFF -> stage4 reproduced exactly. Analytic, not Bellhop.
# ============================================================================
RAY_PHYSICS    = os.environ.get("RAY_PHYSICS", "0") == "1"
SSP_GRADIENT   = float(os.environ.get("SSP_GRADIENT", "-0.2"))   # dc/dz [m/s per m]; <0 = downward-refracting (summer)
WATER_DEPTH    = float(os.environ.get("WATER_DEPTH", "100.0"))   # seabed depth H [m] (AUV at DEPTH_REF sits above it)
REFL_SURFACE   = float(os.environ.get("REFL_SURFACE", "-1.0"))   # surface reflection coeff (pressure-release ~ -1)
REFL_BOTTOM    = float(os.environ.get("REFL_BOTTOM", "0.4"))     # seabed reflection coeff (sediment ~0.3-0.7)
SYMBOL_RATE_HZ = float(os.environ.get("SYMBOL_RATE_HZ", "1000.0"))  # symbol rate for the Doppler penalty


def refract_ray(h, d, c0):
    """Linear SSP c(z)=c0+g*z (g=SSP_GRADIENT) -> ray is a CIRCULAR arc. h=horizontal range, d=AUV depth (>0 down).
    Returns (travel_time_s, elev_bias_rad, ok). ok=False -> no connecting ray (shadow zone). elev_bias = ray
    arrival elevation at the ASV minus the true straight-line elevation (the USBL bearing error)."""
    g = SSP_GRADIENT
    if abs(g) < 1e-9 or h < 1.0:
        return math.hypot(h, d) / c0, 0.0, True          # no gradient / overhead -> straight line
    z_c = -c0 / g                                         # circle-center depth (where c extrapolates to 0)
    r_c = (h*h + d*d - 2.0*d*z_c) / (2.0*h)               # both endpoints equidistant from center -> center r
    R = math.hypot(r_c, z_c)                              # ray radius of curvature
    if R < 1e-6:
        return None, None, False
    cos_s = c0 / (R*abs(g)); cos_r = (c0 + g*d) / (R*abs(g))
    if cos_s > 1.0 or cos_r > 1.0:
        return None, None, False                          # turning point between them -> shadow zone
    th_s = math.acos(min(1.0, cos_s)); th_r = math.acos(min(1.0, cos_r))
    sin_s = min(math.sin(th_s), 0.999999); sin_r = min(math.sin(th_r), 0.999999)
    t = abs(math.atanh(sin_s) - math.atanh(sin_r)) / abs(g)   # exact travel time along the arc
    elev_bias = th_s - math.atan2(d, h)                   # ray elevation at ASV vs straight-line elevation
    return t, elev_bias, True


def multipath_kfactor(h, d):
    """Image-method multipath in the shallow channel (seabed at WATER_DEPTH). Dominant echo = bottom bounce (image
    of the AUV mirrored below the seabed). Returns (rician_K, delay_spread_s): K = direct/echo power ratio (spreading
    + bottom reflection loss), delay_spread = (echo path - direct path)/c."""
    L_d = math.hypot(h, d)                                # direct path length
    L_b = math.hypot(h, 2.0*WATER_DEPTH - d)              # bottom-bounce (AUV image at 2H-d)
    c = sound_speed_mackenzie(TEMP_C, SAL_PPT, d)
    p_direct = 1.0 / (L_d*L_d)
    p_echo   = (REFL_BOTTOM*REFL_BOTTOM) / (L_b*L_b)      # spreading + reflection loss
    K = p_direct / max(p_echo, 1e-12)
    delay_spread = max(0.0, (L_b - L_d)) / c
    return K, delay_spread


def doppler_penalty_db(v_radial, f_khz):
    """Doppler PER penalty [dB of effective SNR]. df = f*v_r/c; penalty grows with (df/symbol_rate)^2 (loss of
    orthogonality). Small at survey speeds (~1 m/s), larger at speed / high frequency."""
    c = sound_speed_mackenzie(TEMP_C, SAL_PPT, 0.0)
    df = (f_khz*1e3) * abs(v_radial) / c
    frac = df / max(SYMBOL_RATE_HZ, 1e-6)
    return 10.0 * math.log10(1.0 + (2.0*frac)**2)         # 0 dB at df=0, grows with the Doppler fraction


def _effective_range(f_khz, sea_state, depth_m=0.0, pd_level=0.5, rmax=5000.0):
    """Range [m] at which p_detect crosses pd_level (linear scan). The physics-based 'max range'."""
    prev = 1.0
    for R in np.arange(1.0, rmax, 1.0):
        pd = p_detect(float(R), f_khz, sea_state, depth_m)
        if pd < pd_level:
            return float(R)
        prev = pd
    return float(rmax)


if PHYSICS_SELFTEST:
    # ---- Stage-A step 1: validate the propagation physics IN ISOLATION vs the published curves, then exit. ----
    _vdir = os.path.join("Results", "acoustic_physics", "validation")
    os.makedirs(_vdir, exist_ok=True)
    _dep = 40.0
    print(f"[selftest] freq={FREQ_KHZ}kHz SL={SL_DB}dB DT={DT_DB}dB DI={DI_DB}dB BW={BW_HZ}Hz "
          f"T={TEMP_C}C S={SAL_PPT}ppt pH={PH_WATER} abs={ABS_MODEL}")
    print(f"[selftest] Mackenzie c(T={TEMP_C},S={SAL_PPT},D={_dep}) = {sound_speed_mackenzie(TEMP_C,SAL_PPT,_dep):.1f} m/s "
          f"(vs constant 1500)")
    print(f"[selftest] alpha @ {FREQ_KHZ}kHz: FG={_alpha_francois_garrison(FREQ_KHZ,depth_m=_dep):.3f}  "
          f"Thorp={_alpha_thorp(FREQ_KHZ):.3f} dB/km")
    for _ss in [0, 2, 4, 6]:
        print(f"[selftest] sea_state={_ss}: wind={wind_speed_from_sea_state(_ss):.1f} m/s  "
              f"NL_psd@{FREQ_KHZ}kHz={ambient_noise_psd_db(FREQ_KHZ,_ss):.1f} dB re uPa^2/Hz  "
              f"| eff.range(P=0.5)={_effective_range(FREQ_KHZ,_ss,_dep):.0f} m  "
              f"| P_detect@150m={p_detect(150.0,FREQ_KHZ,_ss,_dep):.2f}")

    _f = np.logspace(0, 2, 200)                                    # 1 .. 100 kHz
    _R = np.arange(1.0, 1200.0, 2.0)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # (1) absorption vs freq: FG vs Thorp
    ax[0, 0].loglog(_f, [_alpha_francois_garrison(x, depth_m=_dep) for x in _f], 'b-', label='Francois-Garrison 1982')
    ax[0, 0].loglog(_f, [_alpha_thorp(x) for x in _f], 'r--', label='Thorp 1967')
    ax[0, 0].axvline(FREQ_KHZ, color='k', ls=':', lw=0.8)
    ax[0, 0].set_xlabel('frequency [kHz]'); ax[0, 0].set_ylabel('absorption [dB/km]')
    ax[0, 0].set_title('Absorption alpha(f)'); ax[0, 0].grid(True, which='both', alpha=0.3); ax[0, 0].legend()
    # (2) ambient noise psd vs freq for sea states (Wenz / Stojanovic)
    for _ss in [0, 2, 4, 6]:
        ax[0, 1].semilogx(_f, [ambient_noise_psd_db(x, _ss) for x in _f], label=f'sea state {_ss}')
    ax[0, 1].axvline(FREQ_KHZ, color='k', ls=':', lw=0.8)
    ax[0, 1].set_xlabel('frequency [kHz]'); ax[0, 1].set_ylabel('NL [dB re uPa^2/Hz]')
    ax[0, 1].set_title('Ambient noise (Wenz/Stojanovic)'); ax[0, 1].grid(True, which='both', alpha=0.3); ax[0, 1].legend()
    # (3) TL and SNR vs range
    ax[1, 0].plot(_R, [transmission_loss_db(x, FREQ_KHZ, _dep) for x in _R], 'b-', label='TL')
    ax[1, 0].plot(_R, [link_snr_db(x, FREQ_KHZ, SEA_STATE, _dep) for x in _R], 'g-', label=f'SNR (ss={SEA_STATE:g})')
    ax[1, 0].axhline(DT_DB, color='r', ls='--', lw=0.8, label=f'DT={DT_DB}dB')
    ax[1, 0].axvline(150.0, color='k', ls=':', lw=0.8, label='old MAXR=150m')
    ax[1, 0].set_xlabel('range [m]'); ax[1, 0].set_ylabel('dB')
    ax[1, 0].set_title(f'TL & SNR vs range @ {FREQ_KHZ}kHz'); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend()
    # (4) P_detect vs range for sea states  (the SOFT range that replaces the hard cutoff)
    for _ss in [0, 2, 4, 6]:
        ax[1, 1].plot(_R, [p_detect(x, FREQ_KHZ, _ss, _dep) for x in _R], label=f'sea state {_ss}')
    ax[1, 1].axhline(0.5, color='gray', ls=':', lw=0.8)
    ax[1, 1].axvline(150.0, color='k', ls=':', lw=0.8, label='old MAXR=150m')
    ax[1, 1].set_xlabel('range [m]'); ax[1, 1].set_ylabel('P(detect)')
    ax[1, 1].set_title('Soft detection vs range'); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend()
    fig.suptitle(f'Stage-A acoustic physics validation (freq={FREQ_KHZ}kHz, SL={SL_DB}dB, DT={DT_DB}dB)')
    fig.tight_layout()
    _out = os.path.join(_vdir, f"physics_validation_{int(FREQ_KHZ)}kHz.png")
    fig.savefig(_out, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f"[selftest] saved {_out}")

    # ---- DROPOUT validation: BER->PER waterfall + floor, and P_success vs range (Monte-Carlo Rician-averaged) ----
    def _per_nofade(snr_db):
        g = 10.0**(snr_db/10.0); ber = _ber(g); return max(1.0 - (1.0 - ber)**NBITS, PER_FLOOR)
    _rng = np.random.default_rng(0)
    def _psucc_mc(R, ss, n=400):                         # fading-averaged success prob
        return float(np.mean([packet_success_prob(R, FREQ_KHZ, ss, _dep, _rng) for _ in range(n)]))
    _snr = np.linspace(-5, 40, 200)
    print(f"[selftest] DROPOUT: model={DROPOUT_MODEL} mod={MOD} NBITS={NBITS} K={K_FACTOR} floor={PER_FLOOR} "
          f"code_gain={CODE_GAIN_DB}dB")
    for _ss in [0, 4, 6]:
        print(f"[selftest]   sea_state={_ss}: P_success @150m={_psucc_mc(150.0,_ss):.3f}  @180m={_psucc_mc(180.0,_ss):.3f}  "
              f"(mean SNR@150m={link_snr_db(150.0,FREQ_KHZ,_ss,_dep):.0f}dB -> floor-limited)")
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
    _mod_save = MOD
    for _m in ["fsk", "psk"]:
        globals()['MOD'] = _m
        ax2[0].semilogy(_snr, [_per_nofade(s) for s in _snr], label=f'{_m.upper()} (NBITS={NBITS})')
    globals()['MOD'] = _mod_save
    ax2[0].axhline(PER_FLOOR, color='gray', ls=':', label=f'floor={PER_FLOOR}')
    ax2[0].set_xlabel('per-bit SNR [dB]'); ax2[0].set_ylabel('PER'); ax2[0].set_ylim(1e-4, 1.2)
    ax2[0].set_title('Packet-error waterfall (no fade)'); ax2[0].grid(True, which='both', alpha=0.3); ax2[0].legend()
    for _ss in [0, 4, 6]:
        ax2[1].plot(_R, [_psucc_mc(x, _ss, n=120) for x in _R], label=f'sea state {_ss}')
    ax2[1].axvline(150.0, color='k', ls=':', lw=0.8, label='old MAXR=150m')
    ax2[1].set_xlabel('range [m]'); ax2[1].set_ylabel('P(fix succeeds)')
    ax2[1].set_title(f'Success vs range (Rician K={K_FACTOR}, {MOD.upper()})'); ax2[1].grid(alpha=0.3); ax2[1].legend()
    fig2.suptitle(f'Dropout validation (model={DROPOUT_MODEL}, floor={PER_FLOOR}, K={K_FACTOR})')
    fig2.tight_layout()
    _out2 = os.path.join(_vdir, f"dropout_validation_{int(FREQ_KHZ)}kHz.png")
    fig2.savefig(_out2, dpi=120, bbox_inches='tight'); plt.close(fig2)
    print(f"[selftest] saved {_out2}")

    # ---- STAGE C validation: refraction bearing-bias + multipath K/delay vs range/gradient ----
    print(f"[selftest] STAGE C: SSP_GRADIENT={SSP_GRADIENT} /s  WATER_DEPTH={WATER_DEPTH}m  REFL_BOTTOM={REFL_BOTTOM}")
    _c0s = sound_speed_mackenzie(TEMP_C, SAL_PPT, 0.0)
    for _hh in [50.0, 100.0, 150.0, 200.0]:
        _tt, _b, _ok = refract_ray(_hh, _dep, _c0s); _K, _dsp = multipath_kfactor(_hh, _dep)
        _poserr = (math.hypot(_hh, _dep) * abs(math.sin(_b))) if _ok else float('nan')
        print(f"[selftest]   h={_hh:.0f}m d={_dep:.0f}m: bias={math.degrees(_b):+.2f}deg -> pos_err~{_poserr:.1f}m "
              f"| K_multipath={_K:.1f} ({10*math.log10(_K):.0f}dB) delay={_dsp*1000:.0f}ms | shadow={'no' if _ok else 'YES'}")
    _Rr = np.arange(10.0, 400.0, 5.0)
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4.5))
    for _g in [-0.5, -0.2, -0.05, 0.017]:
        _bias = []
        for _x in _Rr:
            _sg = SSP_GRADIENT; globals()['SSP_GRADIENT'] = _g
            _t, _bb, _o = refract_ray(_x, _dep, _c0s); globals()['SSP_GRADIENT'] = _sg
            _bias.append(math.degrees(_bb) if _o else float('nan'))
        ax3[0].plot(_Rr, _bias, label=f'g={_g:+g}/s')
    ax3[0].set_xlabel('horizontal range [m]'); ax3[0].set_ylabel('bearing bias [deg]')
    ax3[0].set_title(f'Refraction bearing bias (AUV depth {_dep:.0f}m)'); ax3[0].grid(alpha=0.3); ax3[0].legend()
    ax3[1].plot(_Rr, [10*math.log10(multipath_kfactor(x, _dep)[0]) for x in _Rr], 'b-', label='Rician K [dB]')
    _ax3b = ax3[1].twinx()
    _ax3b.plot(_Rr, [multipath_kfactor(x, _dep)[1]*1000 for x in _Rr], 'r--', label='delay spread [ms]')
    ax3[1].set_xlabel('horizontal range [m]'); ax3[1].set_ylabel('Rician K [dB]', color='b')
    _ax3b.set_ylabel('delay spread [ms]', color='r')
    ax3[1].set_title(f'Multipath (H={WATER_DEPTH:.0f}m)'); ax3[1].grid(alpha=0.3)
    fig3.suptitle('Stage C validation: refraction + multipath geometry')
    fig3.tight_layout()
    _out3 = os.path.join(_vdir, f"stagec_validation_{int(FREQ_KHZ)}kHz.png")
    fig3.savefig(_out3, dpi=120, bbox_inches='tight'); plt.close(fig3)
    print(f"[selftest] saved {_out3}")
    raise SystemExit(0)

# ASV motion (Otter — direct-thrust scheme)
ASV_CRUISE         = 3.0
ASV_THRUST_PER_MS  = 316.9
ASV_FMAX_THRUSTER  = 1000.0
ASV_KP_V           = 90.0
ASV_KP_YAW         = 450.0
ASV_KD_YAW         = 240.0     # raised 160->240: more yaw-rate damping -> less steering oscillation
ASV_TURN_SLOWDOWN  = True
ASV_LOOKAHEAD      = 11.0      # 11 m: hug the committed plan (16 m cut corners / skipped waypoints).
                              # Smoothness now comes from a straighter PLAN (lower MPPI_SIGMA), not
                              # from a long corner-cutting carrot.
ASV_YAW_PER_N      = 0.001848
V_ASV              = ASV_CRUISE

# Planner geometry
R_KEEPOUT   = 8.0
STANDOFF_R  = 10.0
RING_R      = 20.0

# MPPI (fixA5 original)
USE_MPPI       = True
MPPI_H         = int(os.environ.get("MPPI_H", "8"))   # planning horizon (steps of MPPI_DT). Default 8 = original.
MPPI_DT        = 1.0
# NON-DIMENSIONALIZED horizon (default OFF = fixed MPPI_H). When AUTO_HORIZON=1 the planning horizon is DERIVED from
# the problem scale each replan: H = clamp( HORIZON_K * fleet_reach / (V_ASV*dt), MPPI_H, MPPI_H_MAX ), where
# fleet_reach = farthest ASV->AUV predicted distance (truth-free: from the planned paths + last fixes, NOT truth).
# So the horizon-reach always covers the fleet at ANY scale -> no magic 8 s constant to re-tune. Capped for compute.
AUTO_HORIZON   = os.environ.get("AUTO_HORIZON", "0") == "1"
HORIZON_K      = float(os.environ.get("HORIZON_K", "1.5"))     # dimensionless: horizon-reach = K * fleet reach
MPPI_H_MAX     = int(os.environ.get("MPPI_H_MAX", "50"))       # hard cap on derived H (bounds per-slot compute)
_MPPI_H_LOG    = []                                            # diagnostic: derived H per replan (AUTO_HORIZON)
MPPI_K         = 192
MPPI_SIGMA     = 0.12      # lowered 0.20->0.12: less turn-rate sampling jitter -> straighter,
                          # more trackable plans (the boat can actually follow them faithfully)
MPPI_WMAX      = 0.85
MPPI_WMARGIN   = 0.90
MPPI_VMIN      = float(os.environ.get("MPPI_VMIN", "1.0"))   # min ASV speed; 0 -> ASV can STOP (sit & hold) vs must-move
MPPI_VMAX      = V_ASV
MPPI_SIGMA_V   = 0.7
MPPI_LAMBDA    = 1.0
MPPI_R_CTRL    = float(os.environ.get("MPPI_R_CTRL", "0.10"))   # turn penalty (yaw_rate^2); raise -> straighter, less weave
MPPI_R_ENERGY  = float(os.environ.get("MPPI_R_ENERGY", "0.0"))  # speed/thrust penalty (v^2); 0=off (faithful). >0 -> ASV
#                                                                  moves less/slower when info-gain doesn't justify -> less looping + energy saved
MPPI_DISCOUNT  = 0.95
MPPI_SEED      = 7777
# SLOT_S: min gap between USBL pings (one ping/slot). Env-configurable (default 1.3 = original). Raising it makes
# ping slots SCARCE -> forces genuine contention when several AUVs compete for the one ASV.
SLOT_S         = float(os.environ.get("SLOT_S", 1.3))

# Γ / Coverage / trigger (fixA5 original)
COVERAGE_WEIGHT = float(os.environ.get("COVERAGE_WEIGHT", "40.0"))  # weight of the coverage term; with COST_BOWL this
#                          sets how STEEP the bowl is -> must be large enough that its slope beats the info-gain pull
#                          and is visible within the (short) horizon. Default 40 = original.
COVERAGE_MARGIN = 0.60
# COST_BOWL: reshape the coverage cost from a PLATEAU (max over AUVs of coverage_penalty -- flat/zero within
# COVERAGE_MARGIN*range, so the optimizer gets NO gradient and wanders/parks) into a BOWL (mean over AUVs of
# (dist/OPER_RANGE)^2 -- a smooth paraboloid whose minimum is the fleet centroid, with a downhill slope EVERYWHERE).
# A bowl has a gradient even far from the bottom, so even a short-horizon MPPI feels the pull to the centre -> it
# should cure BOTH the parking (flat cost) AND the myopia (no reachable gradient) with the cheap H=8 planner.
# Default 0 = original plateau EXACTLY.
COST_BOWL = os.environ.get("COST_BOWL", "0") == "1"
SAFEGUARD_S     = 15.0
# USE_REQUEST_TRIGGER: env-configurable (default 1 = current adaptive behaviour). Set 0 to DISABLE the
# covariance trigger so only the SAFEGUARD tier fires -> the ASV pings each AUV every ~SAFEGUARD_S s = a
# FIXED-INTERVAL (periodic) baseline, for the adaptive-vs-periodic comparison.
USE_REQUEST_TRIGGER = os.environ.get("USE_REQUEST_TRIGGER", "1") == "1"
REQUEST_BUDGET  = 20.0
# ============================================================================
# NEW ARCHITECTURE (this file only): ASV-SIDE covariance trigger, no AUV uplink request.
# The AUV never asks for a fix (an acoustic uplink may not arrive). Instead the ASV runs a per-AUV
# SHADOW EKF -- the SAME EKFOOSM class, initialised at the known deploy pose + IMU spec (Q) -- and
# drives it with the PLAN's EXPECTED IMU (level, constant-velocity survey), NOT the AUV's real IMU.
# Covariance propagation P <- F P F^T + Q is DETERMINISTIC (no measurement dependence), so the ASV can
# reproduce each AUV's covariance growth from public knowledge alone. It pings when the shadow's
# position-covariance trace crosses ASV_TRIGGER, and collapses the shadow with the USBL measurement it
# itself makes. ANTI-CHEAT: the shadow reads ONLY constants + its own state + the ASV's own USBL
# measurements -- never auv_truths, never the real EKF (a.ekf.P), never the AUV's real IMU.
USE_ASV_TRIGGER = os.environ.get("USE_ASV_TRIGGER", "1") == "1"
ASV_TRIGGER     = float(os.environ.get("ASV_TRIGGER", 20.0))   # shadow trace(P_pos) threshold (= REQUEST_BUDGET)
# ---- AUV self-check report (piggybacked on the USBL reply; diagnostic ONLY, never triggers) ----
# When the ASV pings, the AUV's reply carries the AUV's OWN real-EKF trace(P_pos) (its belief). The ASV pairs it
# with the shadow trace it SNAPSHOTS at the same emission instant (its own clock -> NO clock sync). The gap =
# "how wrong is my shadow", measured in-field. This never sets req_pending and never selects a target.
USE_AUV_REPORT  = os.environ.get("USE_AUV_REPORT", "1") == "1"
# Optional correction hook (default OFF): if the RUNNING-MEAN discrepancy over the last SHADOW_ADAPT_N reports
# exceeds SHADOW_ADAPT_BAND (m^2), nudge the shadow's position covariance to close the gap. Off because a single
# delayed sample is untrustworthy; act on the TREND only, and only when explicitly enabled.
SHADOW_ADAPT       = os.environ.get("SHADOW_ADAPT", "0") == "1"
SHADOW_ADAPT_N     = int(os.environ.get("SHADOW_ADAPT_N", "5"))
SHADOW_ADAPT_BAND  = float(os.environ.get("SHADOW_ADAPT_BAND", "5.0"))
TRIGGER_OBJECTIVE = "D"
STALENESS_LAMBDA = 0.015
COMMUNICATION_LOSS_PROB = 0.12
COMM_PENALTY_SCALE = 0.5
Q_XY_BASE = 0.02**2

# --- Learned inertial front-end (our TLIO ML) -----------------------------------
# USE_ML: fuse the network's displacement+covariance into each AUV's EKF via stochastic cloning.
# USE_COMPASS: heading aid. First run is IMU-ONLY per project decision -> keep False for a fair
#   classical-vs-learned comparison (both navigate on IMU alone; heading is then gyro-only and drifts).
# ML_MOCK: bypass the network with a stand-in displacement to validate the cloning math alone.
import os as _os_flags
USE_ML       = _os_flags.environ.get("USE_ML", "0") == "1"
USE_COMPASS  = _os_flags.environ.get("USE_COMPASS", "0") == "1"
ML_MOCK      = _os_flags.environ.get("ML_MOCK", "0") == "1"
ML_WIN_TICKS = 50            # non-overlapping window length in sim ticks (50 * 0.02 s = 1.0 s)
# Nonholonomic constraint (AI-IMU / Brossard et al.): a torpedo AUV barely slips sideways, so the
# body-frame LATERAL velocity ~ 0. Fusing that as a pseudo-measurement couples (vx,vy) to psi, making
# heading partially observable from the IMU ALONE (targets the yaw-ambiguous horizontal axis). Still
# IMU-only: no new sensor, just a motion-model assumption.
USE_NHC      = _os_flags.environ.get("USE_NHC", "0") == "1"
NHC_LAT_NOISE = 0.10         # m/s, std of the assumed residual side-slip (how hard to enforce v_lat~0)
# Fuse ONLY the network's calibrated STRONG axis (depth, z) and let the EKF + nonholonomic constraint
# own the yaw-ambiguous horizontal (x,y). Avoids the xy overconfidence seen when fusing the weak axes.
ML_Z_ONLY    = _os_flags.environ.get("ML_Z_ONLY", "0") == "1"
# Course-over-ground heading aid (Phase B): when the AUV is moving and not turning hard, the direction of
# travel ~ heading (small side-slip). Fusing psi ~ atan2(vy,vx) makes yaw observable from motion alone,
# complementing NHC. Gated on speed and turn rate; sigma shrinks with speed. Motion-model only, no sensor.
USE_COG       = _os_flags.environ.get("USE_COG", "0") == "1"
COG_SPEED_MIN = float(_os_flags.environ.get("COG_SPEED_MIN", 0.3))     # m/s, gate + sigma floor
COG_SIG0      = float(_os_flags.environ.get("COG_SIG0", 0.5))          # rad*(m/s): sigma = COG_SIG0/speed
COG_TURN_MAX  = float(_os_flags.environ.get("COG_TURN_MAX", math.radians(8.0)))  # rad/s, skip in turns
# DIAGNOSTIC PROBE (default OFF, not part of the deliverable navigator): forward-speed aid. NHC pins lateral
# velocity and COG pins heading, but NOTHING observes the ALONG-TRACK (forward) speed -- the trace breakdown
# shows velVar_vx ~2 while velVar_vy ~0.02. This aid measures the body-frame forward velocity so we can test
# whether along-track observability is what pins the USBL fix cadence. SPEED_AID_SRC: "dvl" = true speed +
# noise (ordinary sensor simulation, like USBL/IMU); "cmd" = commanded cruise SPEED + noise (NO ground truth).
USE_SPEED_AID   = _os_flags.environ.get("USE_SPEED_AID", "0") == "1"
SPEED_AID_SIGMA = float(_os_flags.environ.get("SPEED_AID_SIGMA", 0.02))   # m/s, DVL-grade ~2 cm/s
SPEED_AID_SRC   = _os_flags.environ.get("SPEED_AID_SRC", "dvl")           # "dvl" | "cmd"
# Calibrated cruise speed for SPEED_AID_SRC="cmd" (an RPM->speed curve measured once on a bench/sea trial;
# truth-free at run time). NOTE: the SPEED constant above (1.0) is a PATH-PLANNING value, not the vehicle's
# actual cruise speed -- at RPM_AUV=1100 the REMUS actually cruises at ~1.83 m/s. Using SPEED here injects a
# -0.83 m/s systematic error into the measurement.
SPEED_AID_CMD   = float(_os_flags.environ.get("SPEED_AID_CMD", 1.83))     # m/s
# DIAGNOSTIC PROBE (default OFF, sidequest only; the pressure sensor stays a v2 item). Depth is NOT observable
# from an IMU -- gravity gives the DIRECTION of down, never the DISPLACEMENT along it -- so with forward speed
# pinned, posVar_z became ~85% of the trace(P) budget. This adds a direct pressure-sensor measurement of z.
USE_DEPTH_AID   = _os_flags.environ.get("USE_DEPTH_AID", "0") == "1"
DEPTH_AID_SIGMA = float(_os_flags.environ.get("DEPTH_AID_SIGMA", 0.1))    # m, typical pressure sensor
# The REMUS inner heading+depth autopilot used to be fed the TRUE eta/nu by mss_env -- i.e. it steered and held
# depth with perfect knowledge no matter how badly the IMU lied. A real AUV closes that loop on what its
# navigation system reports. DEFAULT ON = realistic (autopilot sees the EKF estimate + bias-corrected gyro).
# Set AUTOPILOT_ON_EST=0 to restore the old truth-fed behaviour (needed to reproduce results recorded before
# 2026-07-09: e.g. the control run's 57/56 fixes, xy 2.32/2.10, ANEES 0.79/0.93).
AUTOPILOT_ON_EST = _os_flags.environ.get("AUTOPILOT_ON_EST", "1") == "1"
# FEED_TRUE_PITCH (diagnostic, default 0): with AUTOPILOT_ON_EST=1, override ONLY the pitch fed to the inner autopilot
# with the TRUE pitch (everything else stays estimated). Isolates whether the depth wobble is the noisy est-pitch.
FEED_TRUE_PITCH = _os_flags.environ.get("FEED_TRUE_PITCH", "0") == "1"
# Fix-cadence knobs are env-overridable so a sweep needs no code edits (constants defined above).
# Primary knob = REQUEST_BUDGET (raise -> the covariance trigger fires later -> sparser, covariance-
# gated fixes); SAFEGUARD_S is the secondary floor.
REQUEST_BUDGET = float(_os_flags.environ.get("REQUEST_BUDGET", REQUEST_BUDGET))
SAFEGUARD_S    = float(_os_flags.environ.get("SAFEGUARD_S", SAFEGUARD_S))
# Alternative request trigger: instead of the covariance (trace(P) > REQUEST_BUDGET), fire a fix when the
# EKF estimate's distance to the nearest point on the PLANNED path exceeds PATH_ERR_BUDGET metres. Note the
# controller steers the ESTIMATE onto the plan, so this stays small (rarely trips). Default off = unchanged.
TRIGGER_ON_PATH_ERR = _os_flags.environ.get("TRIGGER_ON_PATH_ERR", "0") == "1"
PATH_ERR_BUDGET     = float(_os_flags.environ.get("PATH_ERR_BUDGET", 20.0))
# Opportunistic Γ-geometry pinging (Tier 3). Default ON = original behaviour. Set 0 for on-demand-only
# pinging (safeguard + request), which is what lets a covariance-driven request reduction cut total fixes.
USE_GAMMA_PING = _os_flags.environ.get("USE_GAMMA_PING", "1") == "1"
# ASV_TIMING: diagnostic only (default OFF). Records the request->receive delay -- from the FIRST tick an AUV
# raises an unanswered request until the tick it actually receives the fix -- which is NOT logged anywhere else
# (log_oosm records only the acoustic round-trip t_arr-t_valid, not the slot wait + contention with the other
# AUV + retries after a lost ping). t_req_raised is STICKY (set only when None) so the delay spans lost-ping
# retries; distinct request CHAINS are counted for a served+unserved==chains cross-check. Pure logging: with the
# flag off, the attributes are still initialised but never read, so no printed number changes.
ASV_TIMING = _os_flags.environ.get("ASV_TIMING", "0") == "1"
# METHOD A -- adaptive TDMA (default OFF). Replaces the greedy tier-2 "ping the highest-trace requester" with a
# need-weighted scheduler: credit = trace(P_xy) * staleness, plus an anti-starvation guarantee (force-serve any
# requester skipped >= TDMA_MAX_SKIP consecutive slots). Only reorders WHICH requester wins a slot; one ping/slot
# unchanged. Flag off reproduces the greedy argmax exactly.
USE_ADAPTIVE_TDMA = _os_flags.environ.get("USE_ADAPTIVE_TDMA", "0") == "1"
TDMA_MAX_SKIP     = int(_os_flags.environ.get("TDMA_MAX_SKIP", "3"))
# METHOD B -- predictive pre-positioning (default OFF). MPPI stays the motion engine; this only scales each AUV's
# info-gain in the MPPI cost by URGENCY = exp(-t_thresh/TAU_URG), where t_thresh is the predicted time until that
# AUV's POS-trace crosses REQUEST_BUDGET (from the same growth rate unc_extrapolated uses). So the ASV drifts
# toward the AUV that will need a fix SOONEST, before the request fires. Truth-free. Flag off -> weight = 1 exactly.
USE_PREDICTIVE_POS = _os_flags.environ.get("USE_PREDICTIVE_POS", "0") == "1"
TAU_URG            = float(_os_flags.environ.get("TAU_URG", "15.0"))
# N_AUVS: number of AUVs (default 2 = original baseline exactly). N_AUVS>=3 adds a 3rd survey lawnmower so the
# single ASV has more demand competing for its one-ping-per-slot -> genuine contention (esp. with SLOT_S raised).
N_AUVS = int(_os_flags.environ.get("N_AUVS", "2"))
# EXTRA_LANES: append this many extra lawnmower lanes to EACH AUV (default 0 = current geometry EXACTLY). Each
# extra lane = one more turn -> larger area covered + longer mission. Extended OUTWARD from the cluster centroid
# (pair NORTH, auv2 SOUTH) so the footprint grows and the single ASV's range coverage is stressed.
EXTRA_LANES = int(_os_flags.environ.get("EXTRA_LANES", "0"))
# AUV_SEP: east offset (m) of AUV1's lawnmower from AUV0's (default 150 = current geometry EXACTLY). Larger =
# surveys further apart -> tests the range envelope (how far can the single ASV serve both). 3rd AUV scales with it.
AUV_SEP = float(_os_flags.environ.get("AUV_SEP", "150"))
# LEG_LEN: lawnmower LEG length [m] (the E-W sweep). Enlarging the survey = lengthen the legs ONLY; leg SPACING
# (60/50 m, the sensor swath) and turn radius (TURN_R) are physical and stay FIXED. LEG_LEN=100 = current geometry
# EXACTLY. With AUV_SEP this sets the E-W footprint (two top lawnmowers side-by-side): E-W span ~= AUV_SEP + LEG_LEN.
LEG_LEN = float(_os_flags.environ.get("LEG_LEN", "100.0"))
# ASV_NAIVE: default 0 = MPPI planner. 1 = BYPASS MPPI, steer the ASV straight to the AUV centroid each slot
# (a trivial baseline). Ping/trigger logic unchanged -> isolates whether the MPPI POSITIONING earns its keep.
ASV_NAIVE = _os_flags.environ.get("ASV_NAIVE", "0") == "1"
# CENTROID_HOLD (ASV_NAIVE only): make the centroid controller SETTLE instead of weaving/orbiting. Default 0 = raw
# centroid (drive at cruise straight to mean(last_fix_xy) each slot) EXACTLY. When 1: (a) low-pass the target so it
# glides instead of hopping on each fix (CENTROID_EMA = EMA factor); (b) speed-schedule so the ASV slows to a crawl
# as it nears the centroid and settles instead of overshooting/orbiting (v=clip(CENTROID_KV*dist, CENTROID_VHOLD,
# ASV_CRUISE)). CENTROID_EMA=1.0 -> no smoothing; CENTROID_KV large -> always cruise.
CENTROID_HOLD  = _os_flags.environ.get("CENTROID_HOLD", "0") == "1"
CENTROID_EMA   = float(_os_flags.environ.get("CENTROID_EMA", "0.15"))   # target low-pass factor (per slot)
CENTROID_KV    = float(_os_flags.environ.get("CENTROID_KV", "0.10"))    # speed per metre of distance-to-centroid
CENTROID_VHOLD = float(_os_flags.environ.get("CENTROID_VHOLD", "0.10")) # min speed when settled (m/s)
# ASYM_MULT / ASYM_AUV: make ONE AUV drift faster (asymmetric fleet). ASYM_MULT scales that AUV's IMU white
# noise (accel+gyro) AND its EKF Q by the same factor -> it stays CALIBRATED (ANEES~1), just noisier -> needs
# disproportionate attention. Default 1.0 = symmetric = baseline EXACTLY. ASYM_AUV picks which AUV (name).
ASYM_MULT = float(_os_flags.environ.get("ASYM_MULT", "1.0"))
ASYM_AUV  = _os_flags.environ.get("ASYM_AUV", "auv0")
# How the MPPI planner combines the per-AUV info-gain when POSITIONING (the ping still picks one AUV/slot).
# "max" (default) = chase the single best AUV -> lunges/weaves between far-apart AUVs. "sum" = reward a
# position decent for BOTH -> central compromise -> far less turning. "min" = help the worst-served AUV
# (fairness). Only changes positioning; default "max" reproduces current behaviour exactly.
MPPI_MULTI_AUV = _os_flags.environ.get("MPPI_MULTI_AUV", "max")
# USBL_ZENITH: the actual USBL horizontal noise now blows up NEAR-OVERHEAD (cone of confusion), matching the
# planner's J model and real USBL:  sigma_h = sl*sa / sin(zenith),  sin(zenith)=h/sl  (clamped at
# ELEV_CONE_MIN_DEG). Default ON. With shallow AUVs + a distant ASV standoff, sin(zenith)~1 -> negligible;
# it only bites when the ASV is nearly above an AUV. Set 0 to recover the old range-only noise.
USBL_ZENITH = _os_flags.environ.get("USBL_ZENITH", "1") == "1"
# Attitude filter mode: "full" = 3D roll/pitch/yaw INS (real gyro, estimated-attitude gravity removal);
# "yaw" = level-locked baseline (roll=pitch=0, yaw-only) for a fair NEES comparison. No truth leak either way.
ATT_MODE     = _os_flags.environ.get("ATT_MODE", "full")
# Depth reference profile amplitude. Default 0 -> constant depth reference held by the REMUS autopilot
# (true depth still has real heave dynamics + tracking error). Set >0 for a sinusoidal depth sweep.
DEPTH_AMP    = float(_os_flags.environ.get("DEPTH_AMP", 0.0))
# Surface-start dive. DIVE_TIME=0 (default) -> AUV spawns at DEPTH_REF (current behaviour). DIVE_TIME>0 ->
# AUV spawns at the surface (z=0) and its commanded depth ramps 0->DEPTH_REF over DIVE_TIME s, then holds.
DIVE_TIME    = float(_os_flags.environ.get("DIVE_TIME", 0.0))
Z_START      = 0.0 if DIVE_TIME > 0.0 else DEPTH_REF

# EKF / AUV sensors  (env-overridable so the accel base can be pinned independently of IMU_GRADE)
IMU_ACC_NOISE      = float(_os_flags.environ.get("IMU_ACC_NOISE", 0.05))
IMU_BIAS_INIT      = float(_os_flags.environ.get("IMU_BIAS_INIT", 0.02))
# Gyro model (real p,q,r fed to BOTH network and EKF). Bias default 0 = today's behaviour; raise
# GYRO_BIAS_INIT for the IMU-drift sweep. Beyond ~0.05 rad/s the network leaves its training augmentation.
GYRO_NOISE_STD     = float(_os_flags.environ.get("GYRO_NOISE_STD", math.radians(0.05)))
GYRO_BIAS_INIT     = float(_os_flags.environ.get("GYRO_BIAS_INIT", 0.0))
GYRO_BIAS_RW       = float(_os_flags.environ.get("GYRO_BIAS_RW", math.radians(0.01)))
# GYRO_BIAS_WALK: model the gyro bias as an in-run RANDOM WALK (drift) instead of a per-run constant.
# When on, the TRUE gyro bias gets a per-tick increment ~N(0, GYRO_BIAS_RW) (std matches the EKF's
# Q[b_g]=GYRO_BIAS_RW^2, so truth+model agree -> calibrated). The drift RATE (GYRO_BIAS_RW) is a known
# datasheet spec ("in-run bias instability"); only the realized path is random. Default 0 = constant bias.
GYRO_BIAS_WALK     = _os_flags.environ.get("GYRO_BIAS_WALK", "0") == "1"
# GYRO_BIAS_RESET_S: if >0, periodically RESET the true gyro bias to 0 AND the EKF's gyro-bias estimate
# (a clean recalibration) every GYRO_BIAS_RESET_S seconds -> bounds heading drift to one interval. The
# per-tick white noise is untouched. Default 0 = never reset = current behaviour exactly.
GYRO_BIAS_RESET_S  = float(_os_flags.environ.get("GYRO_BIAS_RESET_S", 0.0))
# IMU GRADE: a single knob that scales ALL sensor error sources together, exactly as choosing a better
# or worse IMU class does in real life (consumer MEMS -> industrial -> tactical -> navigation-grade;
# every datasheet spec improves together). g<1 = better IMU (less noise+bias), g>1 = worse. The EKF's Q
# is derived from these (grade-scaled) specs below -- standard datasheet practice, NOT ground truth --
# so a better grade -> smaller Q -> slower covariance growth -> fewer covariance-triggered USBL fixes,
# while staying calibrated (verify with ANEES). g=1.0 reproduces the baseline tuning exactly.
IMU_GRADE = float(_os_flags.environ.get("IMU_GRADE", 1.0))
IMU_ACC_NOISE  *= IMU_GRADE
IMU_BIAS_INIT  *= IMU_GRADE
GYRO_NOISE_STD *= IMU_GRADE
GYRO_BIAS_INIT *= IMU_GRADE
GYRO_BIAS_RW   *= IMU_GRADE
# Model-error FLOOR on Q (Option 1): the part of the process noise a better IMU CANNOT remove --
# residual accel bias, strapdown approximation, the NHC/COG side-slip assumption, gravity coupling from
# attitude error. Added on TOP of the grade-scaled sensor spec (in quadrature) so Q saturates at this
# floor instead of shrinking to 0 as IMU_GRADE improves -> the filter stays honest (ANEES~1) at good
# grades instead of going overconfident. Default 0.0 = no floor = reproduces the pre-floor behaviour
# exactly. Tuned empirically so ANEES~1 holds across grades.
Q_VEL_FLOOR = float(_os_flags.environ.get("Q_VEL_FLOOR", 0.0))     # m/s^2-equiv velocity process floor
Q_ATT_FLOOR = float(_os_flags.environ.get("Q_ATT_FLOOR", 0.0))     # rad attitude process floor
# Diagnostic: when TRACE_DIAG=1, print the per-axis position/velocity variance breakdown at each fix
# request, so we can see WHICH state drives trace(P) to REQUEST_BUDGET. Default off (no behaviour change).
TRACE_DIAG  = _os_flags.environ.get("TRACE_DIAG", "0") == "1"
# Accel-bias uncertainty knobs. These feed FORWARD-SPEED (along-track) variance via F[vx,bax], which the
# diagnostic showed dominates trace(P). P0_ABIAS_REG = initial accel-bias std regularizer (how well the
# filter thinks it knows the bias at t0); Q_ABIAS_RW = accel-bias random-walk process std. Defaults reproduce
# the original hardcoded 0.01 / 0.005. Lower them (with a genuinely low, well-known bias) to slow trace(P).
P0_ABIAS_REG = float(_os_flags.environ.get("P0_ABIAS_REG", 0.01))
Q_ABIAS_RW   = float(_os_flags.environ.get("Q_ABIAS_RW", 0.005))
# --- Edit 19 -----------------------------------------------------------------------------------------------
# BUG: the EKF seeds WORLD velocity as (vx=SPEED=1.0, vy=0), but the REMUS spawns at BODY surge 1.5 m/s with
# yaw0=39.8 deg -> true world velocity is 1.5*[cos yaw0, sin yaw0] = [1.152, 0.960]. The seed is therefore wrong
# in magnitude AND direction (|err| = 0.972 m/s, mostly cross-track). Velocity is unobservable, so it integrates
# (97 m by t=100 s) EVEN WITH A PERFECT IMU. EKF_SEED_FIX=1 seeds the correctly-rotated spawn velocity.
# SPAWN_SURGE is a constant WE author (must match mss_env.py:63), not a runtime read of the true state.
SPAWN_SURGE  = float(_os_flags.environ.get("SPAWN_SURGE", 1.5))
EKF_SEED_FIX = _os_flags.environ.get("EKF_SEED_FIX", "0") == "1"
# Initial velocity variance P0[vx,vy,vz]. DANGER: lowering this before EKF_SEED_FIX=1 is catastrophic
# overconfidence (real initial error is 0.972 m/s; sigma=0.01 => NEES ~ 9450).
P0_VEL_INIT  = float(_os_flags.environ.get("P0_VEL_INIT", 0.2))
# Gyro-bias P0 regularizer (was hardcoded radians(0.1) in two places).
P0_GBIAS_REG = float(_os_flags.environ.get("P0_GBIAS_REG", math.radians(0.1)))
# Our injected bias is a CONSTANT draw from N(0, sigma_b) -- it does not random-walk. The correct model is then
# P0[b] = sigma_b^2 and Q[b] = 0, which makes the covariance term EQUAL the real drift statistics exactly:
#     covariance Var(b_a)0 * t^4/4 == reality Var(0.5*b*t^2) = sigma_b^2 * t^4/4
# Default OFF reproduces today's (bias-blind) covariance.  NOTE: honest only because the sim's bias is constant;
# a real IMU wanders, so on hardware keep Q_ABIAS_RW > 0 from the datasheet bias-instability spec.
BIAS_MATCHED_COV = _os_flags.environ.get("BIAS_MATCHED_COV", "0") == "1"
# --- Edit 21: first-order GAUSS-MARKOV bias model (default OFF) -------------------------------------------
# Q[b] must not be a free knob. Q[b]=0 collapses P[b] -> zero gain on a WEAKLY-observable state -> the filter
# locks onto a wrong bias and can never recover (measured: ANEES 510,797, xy 9.2 km). But the hardcoded
# Q_ABIAS_RW=0.005 is equally arbitrary: q_ba*t buries P0[b] 11x within ONE SECOND, which is exactly why the
# bias sweep came out flat. The textbook IMU bias model -- and what a datasheet's Allan variance parameterises:
#     db/dt = -b/tau_b + w   =>   F[b,b] = 1 - dt/tau_b ,  Q[b] = 2*sigma_b^2*dt/tau_b ,  P0[b] = sigma_b^2
# Both P0[b] AND Q[b] then scale with sigma_b^2, so trace(P) tracks the bias knob; and the free-process steady
# state P* = q/(1-a^2) = sigma_b^2 means P[b] can NEVER collapse, so the gain stays alive.
# The two failure modes are the two limits: tau_b -> inf (Q->0, collapse) and tau_b small (Q large, bias-blind).
# HONESTY: our injected bias is EXACTLY constant (tau_b = inf), so a finite tau_b is a deliberate CONSERVATIVE
# mismatch chosen for robustness, not fitted to the sim. Conservative => ANEES <= 1 (the safe direction).
BIAS_GM     = _os_flags.environ.get("BIAS_GM", "0") == "1"
TAU_ABIAS   = float(_os_flags.environ.get("TAU_ABIAS", 300.0))   # s, accel-bias correlation time
TAU_GBIAS   = float(_os_flags.environ.get("TAU_GBIAS", 300.0))   # s, gyro-bias correlation time
BIAS_SIG_EPS = float(_os_flags.environ.get("BIAS_SIG_EPS", 1e-4))  # floor so a PERFECT IMU is not degenerate
# Quasi-static leveling gate (default OFF = legacy behaviour). update_leveling's own docstring: it is valid only
# "when kinematic acceleration is small (cruising AUV)". But LEVEL_INIT_ONLY=1 fires it ONLY for t < 2 s --
# exactly when the AUV accelerates 1.5->1.84 m/s AND starts its dive, i.e. maximum kinematic acceleration -- and
# it never runs again to correct itself. Measured: 58.87 m of injected drift with a PERFECT IMU (vs 0.66 m with
# no aids). LEVEL_GATE=1 instead applies the update only when the assumption actually holds.
LEVEL_GATE      = _os_flags.environ.get("LEVEL_GATE", "0") == "1"
LEVEL_ACC_TOL   = float(_os_flags.environ.get("LEVEL_ACC_TOL", 0.3))              # m/s^2, | |f| - g |
LEVEL_GYRO_TOL  = float(_os_flags.environ.get("LEVEL_GYRO_TOL", math.radians(2.0)))  # rad/s, |omega|
# Launch attitude priors (were hardcoded radians(3) tilt / radians(2) yaw). "Known-start init: the launch pose
# is known (surface GPS fix before diving)" -- so:
#  * TILT is observable from gravity by a STATIONARY surface leveling (at rest, pre-dive: no kinematic accel,
#    which is exactly where update_leveling's assumption is valid). Its accuracy is set by the accelerometer:
#    sigma_tilt = sqrt(acc_noise^2 + acc_bias^2)/g   -> derived when BIAS_MATCHED_COV=1, and -> 0 for a perfect IMU.
#  * YAW is NOT observable from gravity (gravity is yaw-invariant) and there is NO compass in v1. It comes from
#    the surface GPS course-over-ground, i.e. it is a launch-procedure number, never derived from the IMU.
P0_ATT_INIT  = float(_os_flags.environ.get("P0_ATT_INIT", math.radians(3.0)))   # roll/pitch prior, rad
P0_YAW_INIT  = float(_os_flags.environ.get("P0_YAW_INIT", math.radians(2.0)))   # yaw prior, rad (GPS course)
if BIAS_MATCHED_COV:
    # Only the LEGITIMATE half: make P0[b] = sigma_b^2 (the prior IS the bias distribution -- a datasheet
    # number), so trace(P) responds to the bias knob. Q[b] is NOT zeroed here: doing so collapsed P[b], killed
    # the bias gain, and diverged (ANEES 510,797). Q[b] is handled by BIAS_GM (or the legacy random walk).
    P0_ABIAS_REG = 1e-4                     # tiny eps: keeps the Kalman gain alive, P non-singular
    P0_GBIAS_REG = math.radians(0.001)      # tiny eps
    if "P0_ATT_INIT" not in _os_flags.environ:      # explicit env still wins
        _G_LOCAL = 9.81                             # GRAVITY is defined further down; keep in sync
        P0_ATT_INIT = max(math.sqrt(IMU_ACC_NOISE**2 + IMU_BIAS_INIT**2) / _G_LOCAL,
                          math.radians(0.001))      # stationary surface leveling, floored by a tiny eps
# Accelerometer leveling: uses the measured specific force as a gravity reference to keep roll/pitch
# (and accel bias) observable — REQUIRED for the honest strapdown (no true-attitude gravity removal) to
# stay bounded. Noise absorbs the neglected kinematic acceleration of a cruising AUV. Accel-only, no GT.
USE_LEVELING       = _os_flags.environ.get("USE_LEVELING", "1") == "1"
LEVEL_ACC_NOISE    = float(_os_flags.environ.get("LEVEL_ACC_NOISE", 0.5))
# LEVEL_INIT_ONLY: run leveling only for the first LEVEL_INIT_S seconds (establish roll/pitch, TLIO-style),
# then stop — after which nothing corrects roll/pitch. Default 0 = leveling every tick (unchanged).
LEVEL_INIT_ONLY    = _os_flags.environ.get("LEVEL_INIT_ONLY", "0") == "1"
LEVEL_INIT_S       = float(_os_flags.environ.get("LEVEL_INIT_S", 2.0))
COMPASS_NOISE      = math.radians(1.0)
COMPASS_BIAS       = math.radians(0.5)
EKF_INIT_POS_NOISE = 2.0
GRAVITY            = 9.81      # NED +z down; used to re-add gravity after rotating specific force to world
# Non-physical velocity bleed (masks drift). OFF by default now that honest accel-bias states subsume it.
USE_VEL_DAMP       = _os_flags.environ.get("USE_VEL_DAMP", "0") == "1"
EKF_VEL_DAMP       = 0.999

# OOSM history buffer
HIST_MAX_TICKS = 100               # 2 s of history (covers TWTT < 0.4 s)

# ILOS + FF
DELTA_LOS    = 12.0
KI_LOS       = 0.02
FF_DIST      = 10.0
TANGENT_STEP = 6
HEADING_WN   = 0.6

# Mission
N_TICKS_MAX = int(_os_flags.environ.get("N_TICKS_MAX", "30000"))  # sim length in ticks (DT=0.02 -> 30000=600 s). Raise for the bigger 1 km survey.
# Env-settable so ANEES can be checked over several bias draws. A single-seed NEES is ONE realization: a large
# |b| draw is not the same as a modelling bug, so conclusions about ANEES need >1 seed. Default 42 = unchanged.
RNG_SEED    = int(_os_flags.environ.get("RNG_SEED", 42))
RPM_AUV     = 1100.0


# ============================================================================
# Path utilities
# ============================================================================
def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# --- Attitude kinematics (Euler ZYX, body->NED, Fossen convention) ----------------
def _Rzyx(phi, theta, psi):
    """Body->NED rotation R = Rz(psi) Ry(theta) Rx(phi)."""
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    return np.array([
        [cps*cth, cps*sth*sph - sps*cph, cps*sth*cph + sps*sph],
        [sps*cth, sps*sth*sph + cps*cph, sps*sth*cph - cps*sph],
        [-sth,    cth*sph,               cth*cph],
    ])


def _dRzyx(phi, theta, psi):
    """Partials (dR/dphi, dR/dtheta, dR/dpsi) built from elementary rotations and their derivatives."""
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    Rx = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
    Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    Rz = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
    dRx = np.array([[0, 0, 0], [0, -sph, -cph], [0, cph, -sph]])
    dRy = np.array([[-sth, 0, cth], [0, 0, 0], [-cth, 0, -sth]])
    dRz = np.array([[-sps, -cps, 0], [cps, -sps, 0], [0, 0, 0]])
    return Rz @ Ry @ dRx, Rz @ dRy @ Rx, dRz @ Ry @ Rx


def _Tzyx(phi, theta):
    """Euler-ZYX rate transform: [phi_dot, theta_dot, psi_dot]^T = T(phi,theta) @ [p,q,r]^T."""
    cph, sph = math.cos(phi), math.sin(phi)
    cth, tth = math.cos(theta), math.tan(theta)
    return np.array([
        [1.0, sph*tth,  cph*tth],
        [0.0, cph,     -sph],
        [0.0, sph/cth,  cph/cth],
    ])


def _psd(M, floor=1e-6):
    """Symmetrize and clip eigenvalues to `floor` so a covariance stays positive-definite."""
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


def build_dubins_lawnmower(lane_x_min, lane_x_max, lane_ys, R, wp_spacing=1.0):
    wps = []
    n = len(lane_ys)
    for i in range(n):
        ly = lane_ys[i]
        if i % 2 == 0: x0, x1 = lane_x_min, lane_x_max
        else:          x0, x1 = lane_x_max, lane_x_min
        npts = max(2, int(abs(x1 - x0) / wp_spacing) + 1)
        for k in range(npts):
            f = k / (npts - 1)
            wps.append((x0 + f * (x1 - x0), ly))
        if i < n - 1:
            ly_next = lane_ys[i + 1]
            cx = x1
            cy = 0.5 * (ly + ly_next)
            east_turn = (i % 2 == 0)
            narc = max(8, int(math.pi * R / wp_spacing))
            for k in range(1, narc + 1):
                theta = -math.pi / 2 + math.pi * (k / narc)
                px = cx + R * math.cos(theta) if east_turn else cx - R * math.cos(theta)
                py = cy + R * math.sin(theta)
                wps.append((px, py))
    return np.array(wps)


def cumulative_path_length(path):
    seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def path_psi_array(path):
    n = len(path)
    psi = np.zeros(n)
    for i in range(n - 1):
        psi[i] = math.atan2(path[i + 1, 1] - path[i, 1],
                            path[i + 1, 0] - path[i, 0])
    psi[-1] = psi[-2] if n > 1 else 0.0
    return psi


def wrap_path_for_planner(path_xy):
    return {"xy": path_xy,
            "psi": path_psi_array(path_xy),
            "s": cumulative_path_length(path_xy)}


def project_to_path(path, x, y, idx_hint, fwd_window=400):
    lo = max(0, idx_hint - 5)
    hi = min(len(path), idx_hint + fwd_window)
    d2 = (path[lo:hi, 0] - x) ** 2 + (path[lo:hi, 1] - y) ** 2
    j = int(np.argmin(d2))
    return lo + j, math.sqrt(d2[j])


# ============================================================================
# 5-state EKF with OOSM history + retrodiction
# ============================================================================
class EKFOOSM:
    """11-state EKF [x, y, vx, vy, psi, bax, bay, bgz, z, vz, baz] with history buffer for OOSM
    retrodiction. The vertical block (z, vz, baz) is the depth-from-IMU extension: depth is tracked
    by integrating the IMU vertical channel (and corrected by the network's strong dp_z + the USBL
    fix's z), not held constant. It is yaw-decoupled, so it does NOT suffer the heading ambiguity
    that limits x,y.

    The bias states (bax, bay accelerometer; bgz gyro) are the core difference from
    the original 5-state filter and the reason the estimate stays bounded: a constant
    accelerometer offset (the injected imu_bias_*, plus any residual gravity leak)
    would otherwise double-integrate into a 1/2*b*t^2 runaway. The USBL position fixes
    make bax/bay observable (position error and accel bias become correlated through
    the propagation Jacobian), so the *same* fix that corrects position also corrects
    the bias — after which propagation between fixes is drift-free. This mirrors the
    working HoloOcean `propagate` (state [px py vx vy yaw bgz bax bay pz]).

    OOSM workflow when a late-arriving fix lands:
        1. Find history entry j whose time is closest to t_valid
        2. Rewind state/cov to entry j
        3. Fuse the fix at t_valid (in-place USBL update)
        4. Replay all stored IMU samples forward to current time
        5. Update history entries along the way (so future rewinds use new estimate)
    """

    # State layout (15): [x, y, vx, vy, psi, phi, theta, bax, bay, bgx, bgy, bgz, z, vz, baz].
    # Named indices replace magic numbers everywhere so the 15-state layout can't be mis-sliced.
    IX_X, IX_Y, IX_VX, IX_VY, IX_PSI, IX_PHI, IX_THETA = 0, 1, 2, 3, 4, 5, 6
    IX_BAX, IX_BAY, IX_BGX, IX_BGY, IX_BGZ, IX_Z, IX_VZ, IX_BAZ = 7, 8, 9, 10, 11, 12, 13, 14
    NX = 15
    POS_IDX   = [IX_X, IX_Y, IX_Z]            # the three position states (x, y, z)
    CLONE_IDX = [IX_X, IX_Y, IX_Z, IX_PSI]    # Phase-A clone: x, y, z, psi (heading cloned so its
                                              # uncertainty inflates the rotated x,y measurement)

    def __init__(self, x, y, vx, vy, psi, z=DEPTH_REF, vz=0.0, phi=0.0, theta=0.0, noise_scale=1.0):
        # Known-start init: the launch pose is known (surface GPS fix before diving). Biases start at 0.
        self.x = np.zeros(self.NX, dtype=float)
        self.x[self.IX_X] = x;   self.x[self.IX_Y] = y
        self.x[self.IX_VX] = vx; self.x[self.IX_VY] = vy
        self.x[self.IX_PSI] = psi; self.x[self.IX_PHI] = phi; self.x[self.IX_THETA] = theta
        self.x[self.IX_Z] = z;   self.x[self.IX_VZ] = vz
        # Tight P0 consistent with a known launch state (few cm..1 m position, ~1-3 deg attitude).
        self.P = np.diag([
            1.0**2, 1.0**2,                                              # x, y
            P0_VEL_INIT**2, P0_VEL_INIT**2,                              # vx, vy
            P0_YAW_INIT**2,                                             # psi   (surface GPS course; no compass)
            P0_ATT_INIT**2, P0_ATT_INIT**2,                             # phi, theta (stationary surface leveling)
            IMU_BIAS_INIT**2 + P0_ABIAS_REG**2, IMU_BIAS_INIT**2 + P0_ABIAS_REG**2,   # bax, bay (P0 matched to accel-bias std)
            GYRO_BIAS_INIT**2 + P0_GBIAS_REG**2,                               # bgx (matched to turn-on bias
            GYRO_BIAS_INIT**2 + P0_GBIAS_REG**2,                               # bgy  + regularizer; == old
            GYRO_BIAS_INIT**2 + P0_GBIAS_REG**2,                               # bgz  value when GYRO_BIAS_INIT=0)
            1.0**2, P0_VEL_INIT**2, IMU_BIAS_INIT**2 + P0_ABIAS_REG**2,  # z, vz, baz (baz P0 matched)
        ])
        # Process noise: attitude ARW on psi/phi/theta; random-walk on all accel & gyro biases.
        # Velocity & attitude process noise are TIED to the (grade-scaled) IMU spec: velocity from the
        # accel white-noise spec, attitude from the gyro white-noise spec. At IMU_GRADE=1 these equal the
        # original literals (0.05, radians(0.05)); a better grade shrinks them -> P grows slower.
        # velocity/attitude Q = (grade-scaled sensor spec) + (fixed model-error floor), in quadrature.
        _q_vel = (IMU_ACC_NOISE * noise_scale)**2 + Q_VEL_FLOOR**2   # noise_scale>1 for an asymmetric (noisier) AUV
        _q_att = (GYRO_NOISE_STD * noise_scale)**2 + Q_ATT_FLOOR**2
        # Bias process noise. Default: the legacy random-walk literals. BIAS_GM=1: derive from a first-order
        # Gauss-Markov model, Q[b] = 2*sigma_b^2*dt/tau_b, so Q[b] scales with the bias knob AND P[b] has a
        # non-zero steady state (= sigma_b^2), i.e. it can never collapse and kill the gain.
        if BIAS_GM:
            _sa = max(IMU_BIAS_INIT,  BIAS_SIG_EPS)     # eps floor: a PERFECT IMU must not be degenerate
            _sg = max(GYRO_BIAS_INIT, BIAS_SIG_EPS)
            _q_ba = 2.0 * _sa**2 * DT / TAU_ABIAS
            _q_bg = 2.0 * _sg**2 * DT / TAU_GBIAS
        else:
            _q_ba = Q_ABIAS_RW**2
            _q_bg = GYRO_BIAS_RW**2
        self.Q = np.diag([
            Q_XY_BASE, Q_XY_BASE,                                       # x, y
            _q_vel, _q_vel,                                             # vx, vy  (accel spec + floor)
            _q_att,                                                     # psi     (gyro spec + floor)
            _q_att, _q_att,                                             # phi, theta (gyro spec + floor)
            _q_ba, _q_ba,                                               # bax, bay
            _q_bg, _q_bg, _q_bg,                                        # bgx, bgy, bgz
            Q_XY_BASE, _q_vel, _q_ba,                                   # z, vz, baz
        ])
        # History buffer: deque of (t, x, P, imu_used_from_t_to_t+dt) where imu = (ax, ay, az, p, q, r)
        self.hist = collections.deque(maxlen=HIST_MAX_TICKS)
        self.n_oosm_apply = 0
        self.n_oosm_skipped = 0
        # --- Stochastic-cloning state (for the learned-displacement update) ---
        # A clone is a frozen snapshot of pose at a window start. We carry the clone block (P_cc) and the
        # live<->clone cross-covariance (P_lc) so the relative displacement measurement is correct.
        # p_clone/psi_clone use the EKF's OWN estimate at clone time (no ground truth).
        self.clone_active = False
        self.clone = None
        self.p_clone = None        # (3,) cloned position estimate (x, y, z)
        self.psi_clone = 0.0       # cloned yaw estimate (defines the network frame at clone time)
        self.P_cc = None           # clone-block covariance
        self.P_lc = None           # cross-covariance between live state and clone
        self.n_learned_apply = 0

    # --- Core math (no history side-effects) -----------------------------
    def _propagate(self, ax_b, ay_b, az_b, gyro3, dt):
        """Full 3D strapdown propagation (Euler ZYX). Body specific force is de-biased, rotated to the
        world (NED) by the ESTIMATED attitude, gravity re-added, then integrated on all three axes.
        Attitude is integrated from the de-biased body rates [p,q,r]. ATT_MODE='yaw' level-locks
        roll=pitch=0 (yaw-only baseline). No ground truth is used."""
        IX = self
        phi, theta, psi = self.x[IX.IX_PHI], self.x[IX.IX_THETA], self.x[IX.IX_PSI]
        p = gyro3[0] - self.x[IX.IX_BGX]
        q = gyro3[1] - self.x[IX.IX_BGY]
        r = gyro3[2] - self.x[IX.IX_BGZ]
        if ATT_MODE == "yaw":                      # level-locked yaw-only baseline
            phi = 0.0; theta = 0.0; p = 0.0; q = 0.0
            self.x[IX.IX_PHI] = 0.0; self.x[IX.IX_THETA] = 0.0
        f_c = np.array([ax_b - self.x[IX.IX_BAX],
                        ay_b - self.x[IX.IX_BAY],
                        az_b - self.x[IX.IX_BAZ]])
        R = _Rzyx(phi, theta, psi)
        a_world = R @ f_c + np.array([0.0, 0.0, GRAVITY])   # re-add gravity (NED, +z down)
        att_dot = _Tzyx(phi, theta) @ np.array([p, q, r])
        damp = EKF_VEL_DAMP if USE_VEL_DAMP else 1.0
        # Integrate position and velocity uniformly on x, y, z.
        self.x[IX.IX_X] += self.x[IX.IX_VX] * dt + 0.5 * a_world[0] * dt * dt
        self.x[IX.IX_Y] += self.x[IX.IX_VY] * dt + 0.5 * a_world[1] * dt * dt
        self.x[IX.IX_Z] += self.x[IX.IX_VZ] * dt + 0.5 * a_world[2] * dt * dt
        self.x[IX.IX_VX] = (self.x[IX.IX_VX] + a_world[0] * dt) * damp
        self.x[IX.IX_VY] = (self.x[IX.IX_VY] + a_world[1] * dt) * damp
        self.x[IX.IX_VZ] = (self.x[IX.IX_VZ] + a_world[2] * dt) * damp
        # Gauss-Markov bias decay: b <- b*(1 - dt/tau_b). With BIAS_GM off this is a no-op (tau -> inf).
        if BIAS_GM:
            _a_ba = 1.0 - dt / TAU_ABIAS
            _a_bg = 1.0 - dt / TAU_GBIAS
            for _i in (IX.IX_BAX, IX.IX_BAY, IX.IX_BAZ):
                self.x[_i] *= _a_ba
            for _i in (IX.IX_BGX, IX.IX_BGY, IX.IX_BGZ):
                self.x[_i] *= _a_bg
        # Integrate attitude.
        self.x[IX.IX_PHI]   += att_dot[0] * dt
        self.x[IX.IX_THETA] += att_dot[1] * dt
        self.x[IX.IX_PSI]    = wrap_pi(self.x[IX.IX_PSI] + att_dot[2] * dt)

        # --- Jacobian F (15x15) ---
        F = np.eye(self.NX)
        if BIAS_GM:
            # d(b_next)/d(b) = 1 - dt/tau_b. This is what gives P[b] a NON-ZERO steady state
            # P* = q/(1-a^2) = sigma_b^2, so P[b] can never collapse and kill the bias gain.
            for _i in (IX.IX_BAX, IX.IX_BAY, IX.IX_BAZ):
                F[_i, _i] = 1.0 - dt / TAU_ABIAS
            for _i in (IX.IX_BGX, IX.IX_BGY, IX.IX_BGZ):
                F[_i, _i] = 1.0 - dt / TAU_GBIAS
        F[IX.IX_X, IX.IX_VX] = dt; F[IX.IX_Y, IX.IX_VY] = dt; F[IX.IX_Z, IX.IX_VZ] = dt
        if USE_VEL_DAMP:
            F[IX.IX_VX, IX.IX_VX] = damp; F[IX.IX_VY, IX.IX_VY] = damp; F[IX.IX_VZ, IX.IX_VZ] = damp
        vidx = (IX.IX_VX, IX.IX_VY, IX.IX_VZ)
        # velocity <- attitude (dominant new coupling): d(a_world)/d(phi,theta,psi) = (dR/d.)f_c
        dR_dphi, dR_dtheta, dR_dpsi = _dRzyx(phi, theta, psi)
        gphi, gtheta, gpsi = dR_dphi @ f_c, dR_dtheta @ f_c, dR_dpsi @ f_c
        for a_i, vi in enumerate(vidx):
            F[vi, IX.IX_PHI]   = gphi[a_i]   * dt
            F[vi, IX.IX_THETA] = gtheta[a_i] * dt
            F[vi, IX.IX_PSI]   = gpsi[a_i]   * dt
            # velocity <- accel-bias: d(a_world)/d(b_a) = -R
            F[vi, IX.IX_BAX] = -R[a_i, 0] * dt
            F[vi, IX.IX_BAY] = -R[a_i, 1] * dt
            F[vi, IX.IX_BAZ] = -R[a_i, 2] * dt
        # attitude <- gyro-bias: d(att_dot)/d(b_g) = -T(phi,theta)
        T = _Tzyx(phi, theta)
        for e_i, oi in enumerate((IX.IX_PHI, IX.IX_THETA, IX.IX_PSI)):
            F[oi, IX.IX_BGX] = -T[e_i, 0] * dt
            F[oi, IX.IX_BGY] = -T[e_i, 1] * dt
            F[oi, IX.IX_BGZ] = -T[e_i, 2] * dt
        self.F = F                          # exposed for the finite-difference Jacobian self-test
        self.P = F @ self.P @ F.T + self.Q
        # Stochastic cloning: the clone block is frozen; the live<->clone cross-cov rides the same F.
        if self.clone_active:
            self.P_lc = F @ self.P_lc

    def clone_pose(self):
        """Snapshot the current 3D position AND heading as the clone for the next learned window.
        Cloning psi (not just position) is what lets heading uncertainty correctly inflate the rotated
        x,y displacement measurement (mirrors TLIO's full-pose clone)."""
        ci = self.CLONE_IDX                        # [0, 1, 8, 4] = x, y, z, psi
        self.clone = self.x[ci].copy()             # (4,) [x, y, z, psi]
        self.p_clone = self.clone[0:3]             # view: cloned position (x, y, z)
        self.psi_clone = float(self.clone[3])      # cloned heading (defines the net frame)
        self.P_cc = self.P[np.ix_(ci, ci)].copy()  # (4,4) clone block (incl. heading variance)
        self.P_lc = self.P[:, ci].copy()           # (11,4): cross between all live states and the clone
        self.clone_active = True

    def update_learned_displacement(self, dp_net_xyz, Sigma_xyz, z_only=False):
        """Fuse a network 3D displacement Δp (over [clone, now], in the network/gravity-aligned frame)
        as a relative-position measurement: z = [Rz(psi_clone)^T (p_xy_now - p_xy_clone); z_now - z_clone]
        = Δp_net, R = Σ. The x,y part rotates by the cloned yaw; z is yaw-invariant (gravity-aligned ==
        world vertical). Uses stochastic cloning (augmented live+clone covariance). No ground truth.

        z_only=True fuses ONLY the depth component (the network's calibrated, yaw-decoupled STRONG axis),
        leaving x,y to the EKF + nonholonomic constraint — avoids the xy overconfidence from the weak axes."""
        if not self.clone_active:
            return False
        if z_only:
            return self._update_learned_z(dp_net_xyz, Sigma_xyz)
        c, s = math.cos(self.psi_clone), math.sin(self.psi_clone)
        RzT = np.array([[c, s], [-s, c]])                 # R(psi_clone)^T  (world -> net frame), x,y only
        dRzT = np.array([[-s, c], [-c, -s]])              # d RzT / d psi_clone
        p_now = self.x[self.POS_IDX]                       # (3,) [x, y, z]
        d_world = p_now - self.p_clone                     # (3,) world-frame displacement
        pred = np.array([RzT[0, 0] * d_world[0] + RzT[0, 1] * d_world[1],
                         RzT[1, 0] * d_world[0] + RzT[1, 1] * d_world[1],
                         d_world[2]])                      # net-frame predicted displacement (z direct)
        nu = np.asarray(dp_net_xyz, float) - pred          # innovation (3,)

        nx, na = self.NX, self.NX + 4                      # 11 live + 4 clone (x,y,z,psi) = 15
        # Augmented covariance Pa = [[P (11x11), P_lc (11x4)], [P_lc^T, P_cc (4x4)]] -> 15x15
        Pa = np.zeros((na, na))
        Pa[0:nx, 0:nx] = self.P
        Pa[0:nx, nx:na] = self.P_lc
        Pa[nx:na, 0:nx] = self.P_lc.T
        Pa[nx:na, nx:na] = self.P_cc
        # Jacobian H (3 x na). Clone cols: nx=x, nx+1=y, nx+2=z, nx+3=psi.
        H = np.zeros((3, na))
        H[0:2, 0:2] = RzT                       # xy meas wrt live x,y
        H[2, self.IX_Z] = 1.0                   # z  meas wrt live z
        H[0:2, nx:nx + 2] = -RzT                # xy meas wrt clone x,y
        H[2, nx + 2] = -1.0                     # z  meas wrt clone z
        H[0:2, nx + 3] = dRzT @ d_world[0:2]    # xy meas wrt clone psi (heading uncertainty inflates xy)
        R = _psd(Sigma_xyz, floor=(1e-3)**2)
        S = H @ Pa @ H.T + R
        K = Pa @ H.T @ np.linalg.inv(S)                   # (na,3)
        dx = (K @ nu).ravel()
        self.x = self.x + dx[0:nx]                          # correct live state (clone corr discarded)
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        Pa = (np.eye(na) - K @ H) @ Pa
        self.P = 0.5 * (Pa[0:nx, 0:nx] + Pa[0:nx, 0:nx].T)  # marginalize back to the live NX x NX
        self.clone_active = False                          # consume clone; caller re-clones
        self.n_learned_apply += 1
        return True

    def _update_learned_z(self, dp_net_xyz, Sigma_xyz):
        """Depth-only variant: fuse just dp_net_z = z_now - z_clone (yaw-invariant) via stochastic
        cloning. R = Sigma_zz. x,y are left untouched (owned by the EKF + nonholonomic constraint)."""
        nx, na = self.NX, self.NX + 4                      # 11 live + 4 clone (x,y,z,psi) = 15
        pred = self.x[self.IX_Z] - self.p_clone[2]         # predicted depth displacement
        nu = float(dp_net_xyz[2]) - pred                   # scalar innovation
        Pa = np.zeros((na, na))
        Pa[0:nx, 0:nx] = self.P
        Pa[0:nx, nx:na] = self.P_lc
        Pa[nx:na, 0:nx] = self.P_lc.T
        Pa[nx:na, nx:na] = self.P_cc
        H = np.zeros((1, na))
        H[0, self.IX_Z] = 1.0                              # live z
        H[0, nx + 2] = -1.0                                # clone z (z is yaw-invariant -> no psi term)
        R = np.array([[max(float(Sigma_xyz[2, 2]), (1e-3)**2)]])
        S = H @ Pa @ H.T + R
        K = Pa @ H.T @ np.linalg.inv(S)                    # (na,1)
        dx = (K * nu).ravel()
        self.x = self.x + dx[0:nx]
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        Pa = (np.eye(na) - K @ H) @ Pa
        self.P = 0.5 * (Pa[0:nx, 0:nx] + Pa[0:nx, 0:nx].T)
        self.clone_active = False
        self.n_learned_apply += 1
        return True

    def _update_usbl(self, x_meas, y_meas, z_meas, R_3x3):
        H = np.zeros((3, self.NX)); H[0, self.IX_X] = 1.0; H[1, self.IX_Y] = 1.0; H[2, self.IX_Z] = 1.0
        nu = np.array([x_meas, y_meas, z_meas]) - H @ self.x
        S = H @ self.P @ H.T + R_3x3
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ nu).ravel()
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        self.P = self._joseph(K, H, R_3x3)

    def _joseph(self, K, H, R):
        """Joseph-form covariance update: P = (I-KH)P(I-KH)^T + KRK^T, symmetrized. Numerically stable
        and keeps P positive-definite over long fix sequences."""
        IKH = np.eye(self.NX) - K @ H
        P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return 0.5 * (P + P.T)

    # --- Public API ------------------------------------------------------
    def predict(self, ax_b, ay_b, az_b, gyro3, dt, t_now):
        """Record (t, x, P, imu=(ax,ay,az,p,q,r)) into history, then propagate."""
        self.hist.append((t_now, self.x.copy(), self.P.copy(),
                          (ax_b, ay_b, az_b, float(gyro3[0]), float(gyro3[1]), float(gyro3[2]))))
        self._propagate(ax_b, ay_b, az_b, gyro3, dt)

    def reset_gyro_bias(self):
        """Hard-reset the gyro-bias states to 0 (a clean recalibration): zero the estimate, clear the
        cross-covariances with all other states, and set the bias variance back to its initial value.
        Only invoked when GYRO_BIAS_RESET_S > 0."""
        bg = [self.IX_BGX, self.IX_BGY, self.IX_BGZ]
        for i in bg:
            self.x[i] = 0.0
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = P0_GBIAS_REG**2           # back to P0[bg]
        self.P = 0.5 * (self.P + self.P.T)

    def update_compass(self, psi_meas):
        H = np.zeros((1, self.NX)); H[0, self.IX_PSI] = 1.0
        R = np.array([[COMPASS_NOISE**2]])
        nu = np.array([wrap_pi(psi_meas - self.x[self.IX_PSI])])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ nu).ravel()
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        self.P = self._joseph(K, H, R)

    def update_nonholonomic(self, sigma_lat=NHC_LAT_NOISE):
        """Nonholonomic side-slip constraint (AI-IMU style): body-frame lateral velocity ~ 0.
        Measurement h(x) = -sin(psi)*vx + cos(psi)*vy (the velocity component perpendicular to
        heading); we fuse it toward 0. The psi-dependence of H couples velocity to yaw, so when the
        AUV is moving the heading becomes partially observable from the IMU alone. No ground truth,
        no new sensor — purely a motion-model assumption. Keeps the clone cross-cov consistent."""
        psi = self.x[self.IX_PSI]; vx, vy = self.x[self.IX_VX], self.x[self.IX_VY]
        c, s = math.cos(psi), math.sin(psi)
        h = -s * vx + c * vy                       # body-frame lateral velocity
        H = np.zeros((1, self.NX))
        H[0, self.IX_VX] = -s
        H[0, self.IX_VY] = c
        H[0, self.IX_PSI] = -c * vx - s * vy       # d(h)/d(psi)
        R = np.array([[sigma_lat**2]])
        nu = np.array([0.0 - h])                   # drive lateral velocity toward zero
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(self.NX) - K @ H
        Plc_new = IKH @ self.P_lc if self.clone_active else None
        self.x = self.x + (K @ nu).ravel()
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        P = IKH @ self.P @ IKH.T + K @ R @ K.T     # Joseph form
        self.P = 0.5 * (P + P.T)
        # A clone is usually active across the window; keep its cross-cov consistent with this update.
        if self.clone_active:
            self.P_lc = Plc_new

    def update_forward_speed(self, v_meas, sigma=SPEED_AID_SIGMA):
        """DIAGNOSTIC forward-speed aid: the exact mirror of update_nonholonomic(). NHC measures the body-frame
        LATERAL velocity h = -sin(psi)*vx + cos(psi)*vy and drives it to 0; here we measure the body-frame
        FORWARD velocity h = cos(psi)*vx + sin(psi)*vy and fuse it toward v_meas. This is the one velocity
        component nothing else observes (NHC pins lateral, COG pins heading DIRECTION but not speed), so it is
        what lets us test whether along-track observability pins the fix cadence. Keeps the clone cross-cov
        consistent, same Joseph form as NHC."""
        psi = self.x[self.IX_PSI]; vx, vy = self.x[self.IX_VX], self.x[self.IX_VY]
        c, s = math.cos(psi), math.sin(psi)
        h = c * vx + s * vy                        # body-frame forward velocity
        H = np.zeros((1, self.NX))
        H[0, self.IX_VX] = c
        H[0, self.IX_VY] = s
        H[0, self.IX_PSI] = -s * vx + c * vy       # d(h)/d(psi)
        R = np.array([[sigma**2]])
        nu = np.array([v_meas - h])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(self.NX) - K @ H
        Plc_new = IKH @ self.P_lc if self.clone_active else None
        self.x = self.x + (K @ nu).ravel()
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        P = IKH @ self.P @ IKH.T + K @ R @ K.T     # Joseph form
        self.P = 0.5 * (P + P.T)
        if self.clone_active:
            self.P_lc = Plc_new

    def update_depth(self, z_meas, sigma=DEPTH_AID_SIGMA):
        """DIAGNOSTIC depth (pressure-sensor) aid: a direct scalar measurement of the z state. Hydrostatic
        pressure gives an ABSOLUTE depth reference, which the IMU can never supply (gravity fixes the direction
        of down, not the displacement along it). NOTE z is in CLONE_IDX, so the clone cross-covariance must be
        maintained, exactly as update_nonholonomic does."""
        H = np.zeros((1, self.NX)); H[0, self.IX_Z] = 1.0
        R = np.array([[sigma**2]])
        nu = np.array([z_meas - self.x[self.IX_Z]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(self.NX) - K @ H
        Plc_new = IKH @ self.P_lc if self.clone_active else None
        self.x = self.x + (K @ nu).ravel()
        P = IKH @ self.P @ IKH.T + K @ R @ K.T     # Joseph form
        self.P = 0.5 * (P + P.T)
        if self.clone_active:
            self.P_lc = Plc_new

    def update_cog(self, yaw_rate=0.0):
        """Course-over-ground heading aid: fuse the soft constraint psi - atan2(vy,vx) ~ 0 (heading equals
        direction of travel when side-slip is small). Couples psi to velocity, so yaw becomes observable
        from motion. Gated on speed (>COG_SPEED_MIN) and turn rate (|yaw_rate|<COG_TURN_MAX). No sensor."""
        vx, vy = self.x[self.IX_VX], self.x[self.IX_VY]
        speed = math.hypot(vx, vy)
        if speed < COG_SPEED_MIN or abs(yaw_rate) > COG_TURN_MAX:
            return
        s2 = vx*vx + vy*vy
        c = wrap_pi(self.x[self.IX_PSI] - math.atan2(vy, vx))     # constraint, driven toward 0
        H = np.zeros((1, self.NX))
        H[0, self.IX_PSI] = 1.0
        H[0, self.IX_VX]  = vy / s2
        H[0, self.IX_VY]  = -vx / s2
        sigma = COG_SIG0 / max(speed, COG_SPEED_MIN)
        R = np.array([[sigma**2]])
        nu = np.array([-c])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(self.NX) - K @ H
        Plc_new = IKH @ self.P_lc if self.clone_active else None
        self.x = self.x + (K @ nu).ravel()
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)
        if self.clone_active:
            self.P_lc = Plc_new

    def update_leveling(self, ax_raw, ay_raw, az_raw, sigma=LEVEL_ACC_NOISE):
        """Accelerometer leveling. When kinematic acceleration is small (cruising AUV), the measured
        specific force ~ gravity-reaction + bias: f_meas = R(phi,theta)^T[0,0,-g] + b_a. Fusing that
        observes roll, pitch, and accel bias from the gravity vector (yaw-invariant). The neglected
        kinematic acceleration is absorbed by `sigma`. Accelerometer-only, no ground truth."""
        if ATT_MODE == "yaw":
            return
        phi, theta = self.x[self.IX_PHI], self.x[self.IX_THETA]
        g = GRAVITY
        cph, sph = math.cos(phi), math.sin(phi)
        cth, sth = math.cos(theta), math.sin(theta)
        h_grav = np.array([g*sth, -g*cth*sph, -g*cth*cph])   # R^T[0,0,-g], depends on phi,theta only
        b = self.x[[self.IX_BAX, self.IX_BAY, self.IX_BAZ]]
        h = h_grav + b
        nu = np.array([ax_raw, ay_raw, az_raw]) - h
        H = np.zeros((3, self.NX))
        H[:, self.IX_PHI]   = [0.0,    -g*cth*cph,  g*cth*sph]
        H[:, self.IX_THETA] = [g*cth,   g*sth*sph,  g*sth*cph]
        H[0, self.IX_BAX] = 1.0; H[1, self.IX_BAY] = 1.0; H[2, self.IX_BAZ] = 1.0
        R = np.eye(3) * sigma**2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(self.NX) - K @ H
        Plc_new = IKH @ self.P_lc if self.clone_active else None
        self.x = self.x + (K @ nu).ravel()
        self.x[self.IX_PSI] = wrap_pi(self.x[self.IX_PSI])
        P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)
        if self.clone_active:
            self.P_lc = Plc_new

    def apply_fix_oosm(self, t_valid, x_meas, y_meas, z_meas, R_3x3, dt):
        """OOSM rewind-fuse-replay. Returns True on success, False if t_valid
        is older than our history window (fix dropped)."""
        # A USBL fix rewinds/replays the filter; the current clone's cross-cov would no longer be
        # consistent, so drop it and let the main loop re-clone fresh next window.
        self.clone_active = False
        if len(self.hist) == 0:
            # Cold start: no history → apply at current time as a fallback
            self._update_usbl(x_meas, y_meas, z_meas, R_3x3)
            self.n_oosm_apply += 1
            return True
        times = np.array([h[0] for h in self.hist])
        if t_valid < times[0] - 0.5:
            # Older than our buffer → drop the fix
            self.n_oosm_skipped += 1
            return False
        # Find closest entry (use side='right' then step back to be ≤ t_valid)
        j = int(np.searchsorted(times, t_valid, side='right') - 1)
        j = max(0, min(j, len(self.hist) - 1))
        # 1) Rewind
        self.x = self.hist[j][1].copy()
        self.P = self.hist[j][2].copy()
        # 2) Fuse at t_j (≈ t_valid)
        self._update_usbl(x_meas, y_meas, z_meas, R_3x3)
        # 3) Replay forward: re-propagate using each stored IMU sample. Overwrite the entry at k=j too
        #    with the POST-fix state, so a second fix landing in the same window rewinds to the corrected
        #    state instead of the stale pre-fix one.
        hist_list = list(self.hist)         # snapshot
        for k in range(j, len(hist_list)):
            t_k, _, _, imu = hist_list[k]
            hist_list[k] = (t_k, self.x.copy(), self.P.copy(), imu)
            self._propagate(imu[0], imu[1], imu[2], np.array(imu[3:6]), dt)
        # Push updated history back into the deque
        self.hist = collections.deque(hist_list, maxlen=HIST_MAX_TICKS)
        self.n_oosm_apply += 1
        return True


# ============================================================================
# Finite-difference Jacobian self-test (run with JACOBIAN_TEST=1; exits before the mission).
# Verifies the analytic propagation Jacobian F against central differences of the state update.
# The two intentionally-dropped 2nd-order blocks (pos<-attitude ~½dt², attitude<-attitude ~|w|dt) are
# below the tolerance; any first-order sign/magnitude error is far above it.
# ============================================================================
def _jacobian_selftest():
    import sys
    IX = EKFOOSM
    ang = {IX.IX_PSI, IX.IX_PHI, IX.IX_THETA}
    dt = 0.01
    # Nominal state with non-trivial attitude, velocity, and biases.
    x0 = np.zeros(IX.NX)
    x0[IX.IX_VX], x0[IX.IX_VY], x0[IX.IX_VZ] = 1.2, -0.3, 0.1
    x0[IX.IX_PSI], x0[IX.IX_PHI], x0[IX.IX_THETA] = 0.30, 0.10, -0.15
    x0[IX.IX_BAX], x0[IX.IX_BAY], x0[IX.IX_BAZ] = 0.02, -0.01, 0.015
    x0[IX.IX_BGX], x0[IX.IX_BGY], x0[IX.IX_BGZ] = 0.005, -0.003, 0.004
    imu = (0.4, -0.2, 9.6 + 0.3, 0.05, -0.02, 0.03)   # ax, ay, az(≈g+heave), p, q, r

    def prop(x):
        e = EKFOOSM(0, 0, 0, 0, 0)
        e.x = x.copy(); e.P = np.zeros((IX.NX, IX.NX)); e.clone_active = False
        e._propagate(imu[0], imu[1], imu[2], np.array(imu[3:6]), dt)
        return e.x.copy()

    e0 = EKFOOSM(0, 0, 0, 0, 0)
    e0.x = x0.copy(); e0.P = np.zeros((IX.NX, IX.NX)); e0.clone_active = False
    e0._propagate(imu[0], imu[1], imu[2], np.array(imu[3:6]), dt)
    F_an = e0.F.copy()

    eps = 1e-6
    F_num = np.zeros((IX.NX, IX.NX))
    for j in range(IX.NX):
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        d = prop(xp) - prop(xm)
        for r in ang:
            d[r] = wrap_pi(d[r])
        F_num[:, j] = d / (2 * eps)

    err = np.abs(F_an - F_num)
    i, j = np.unravel_index(np.argmax(err), err.shape)
    print(f"[jacobian-test] max|F_analytic - F_numeric| = {err.max():.2e} at (row={i}, col={j})")
    tol = 1e-2
    ok = err.max() < tol
    print(f"[jacobian-test] {'PASS' if ok else 'FAIL'} (tol={tol}, dropped 2nd-order blocks are below tol)")
    sys.exit(0 if ok else 1)


if _os_flags.environ.get("JACOBIAN_TEST", "0") == "1":
    _jacobian_selftest()


# ============================================================================
# ───────────── PORTED VERBATIM FROM Controller_test.py ─────────────
# ============================================================================
def usbl(asv, auv, t, rng, asv_vel=None, auv_vel=None):
    dx = auv[0] - asv[0]; dy = auv[1] - asv[1]; dz = auv[2]
    h  = np.hypot(dx, dy); sl = np.hypot(h, dz)
    depth_m = abs(dz)
    # --- Stage C: propagation GEOMETRY (refraction bending + multipath + Doppler). Truth-free: acts on the signal
    # like the true range does; no ASV decision reads truth. Off -> ray_bias=0, straight travel time, fixed K. ---
    ray_bias = 0.0; ray_tt = None; k_geo = None; dop_pen = 0.0
    if RAY_PHYSICS and not PHYSICS_OFF:
        _c0 = sound_speed_mackenzie(TEMP_C, SAL_PPT, 0.0)
        ray_tt, ray_bias, _ok = refract_ray(h, depth_m, _c0)
        if not _ok:
            return None                                       # shadow zone: no connecting ray -> no fix
        k_geo, _ = multipath_kfactor(h, depth_m)              # Rician K from the bottom-bounce geometry
        if asv_vel is not None and auv_vel is not None and sl > 1e-6:
            _u = np.array([dx, dy, dz]) / sl
            _av = np.asarray(asv_vel, float); _bv = np.asarray(auv_vel, float)
            _rel = np.array([_av[0]-_bv[0], _av[1]-_bv[1],
                             (_av[2] if _av.size > 2 else 0.0) - (_bv[2] if _bv.size > 2 else 0.0)])
            dop_pen = doppler_penalty_db(float(np.dot(_rel, _u)), FREQ_KHZ)
    # --- Detection: physics (sonar eq + Rayleigh-fading soft P_detect) vs old hard-cutoff (PHYSICS_OFF) ---
    if PHYSICS_OFF:
        if sl > MAXR or rng.random() < LOSS:
            return None
        crlb_scale = 1.0
    else:
        # Soft detection, DROPOUT_MODEL: 'per' = realistic BER->PER + Rician fade + floor; 'rayleigh' = Stage-A
        # outage; 'none' = perfect link (detects at any range). No hard MAXR -> range emerges from SNR.
        if DROPOUT_MODEL == "none":
            pd = 1.0
        elif DROPOUT_MODEL == "rayleigh":
            pd = p_detect(sl, FREQ_KHZ, SEA_STATE, depth_m)
        else:
            pd = packet_success_prob(sl, FREQ_KHZ, SEA_STATE, depth_m, rng, k=k_geo, snr_penalty_db=dop_pen)
        if math.isfinite(SHALLOW_MAXR):
            # Shallow-water operational-range limit (reverberation, absent from the free-field sonar eq): the cheap
            # device's rated envelope. Soft logistic rolloff centered at SHALLOW_MAXR over ~5% width -> ~1 at short
            # range, 0.5 at the rating, ->0 beyond. Uses the TRUE slant range (physics), never an ASV decision.
            pd *= 1.0 / (1.0 + math.exp((sl - SHALLOW_MAXR) / (0.05 * SHALLOW_MAXR)))
        if rng.random() > pd:
            return None
        snr_db = link_snr_db(sl, FREQ_KHZ, SEA_STATE, depth_m)
        # CRLB: range/bearing sigmas grow ~1/sqrt(SNR) as SNR falls toward threshold (clamped >=1x at high SNR).
        crlb_scale = max(1.0, math.sqrt(10.0**((SNR_REF_DB - snr_db) / 10.0)))
    sig_r_eff = SIG_R * crlb_scale
    sa = np.radians(SIG_A) * crlb_scale
    rm = sl + rng.normal(0, sig_r_eff)
    az = np.arctan2(dy, dx); el = np.arctan2(dz, max(h, 1e-6))
    # Stage C: the ray arrives at the ASV at a bent elevation -> the USBL mis-reads the bearing by ray_bias,
    # placing the fix at the wrong elevation (a systematic position error). 0 when RAY_PHYSICS off.
    azm = az + rng.normal(0, sa); elm = el + ray_bias + rng.normal(0, sa)
    hm = rm * np.cos(elm); zm = rm * np.sin(elm)
    nx = asv[0] + hm * np.cos(azm) + rng.normal(0, SIG_GPS)
    ny = asv[1] + hm * np.sin(azm) + rng.normal(0, SIG_GPS)
    nz = zm
    if rng.random() < OUT_P:
        nx += rng.normal(0, OUT_B); ny += rng.normal(0, OUT_B); nz += rng.normal(0, OUT_B)
    ang_h = sl * sa
    if USBL_ZENITH:                                  # cone of confusion: worse horizontal fix near-overhead
        _sinz = max(h / max(sl, 1e-6), math.sin(math.radians(ELEV_CONE_MIN_DEG)))
        ang_h = sl * sa / _sinz
    sh = np.sqrt(ang_h**2 + sig_r_eff**2 + SIG_GPS**2)
    sz = np.sqrt((sl * sa)**2 + sig_r_eff**2)
    # Sound speed: Mackenzie(T,S,depth) in physics mode (real c ~1490 at 10C/35ppt/40m), constant 1500 if OFF.
    c_sound  = C_SOUND if PHYSICS_OFF else sound_speed_mackenzie(TEMP_C, SAL_PPT, depth_m)
    one_way  = ray_tt if (ray_tt is not None) else sl / c_sound   # Stage C: refracted arc travel time (else straight)
    if MAC_TIMING:
        # Realistic MAC cycle: telegram TX + transponder turn-around + guard, on top of propagation.
        t_valid  = t + T_PING + one_way          # AUV pos captured when the interrogation arrives
        t_arrive = t + usbl_cycle_time(sl, depth_m)
    else:
        t_valid  = t + one_way
        t_arrive = t + 2.0 * one_way + PROC + (one_way if DOWNLINK_LEG else 0.0)
    return (t, t_arrive, np.array([nx, ny, nz]),
            np.diag([sh**2, sh**2, sz**2]), t_valid)


def standoff(asv, auv, Rk):
    v = asv - auv; dd = np.hypot(*v)
    u = v / dd if dd > 1e-6 else np.array([1., 0.])
    p = auv + Rk * u
    return np.array([p[0], p[1], np.arctan2(auv[1] - p[1], auv[0] - p[0])])


def usbl_geometry(asv_xy, auv_xyz):
    dx = auv_xyz[0] - asv_xy[0]; dy = auv_xyz[1] - asv_xy[1]; dz = abs(auv_xyz[2])
    h = np.hypot(dx, dy); sl = np.hypot(h, dz); brg = math.atan2(dy, dx)
    return sl, brg


def J_meas_from_geometry(asv_xy, auv_xyz):
    sl, brg = usbl_geometry(asv_xy, auv_xyz)
    sa = np.radians(SIG_A)
    h = np.hypot(auv_xyz[0] - asv_xy[0], auv_xyz[1] - asv_xy[1])
    sin_zen = max(h / max(sl, 1e-6), math.sin(math.radians(ELEV_CONE_MIN_DEG)))
    sigma_r   = SIG_R
    sigma_ang = max(sl * sa / sin_zen, 1e-3)
    V = np.array([[math.cos(brg), -math.sin(brg)],
                  [math.sin(brg),  math.cos(brg)]])
    Lam = np.diag([sigma_r**2, sigma_ang**2])
    Rm = V @ Lam @ V.T + (SIG_GPS**2) * np.eye(2)
    return np.linalg.inv(Rm), sl


def predict_on_path(path, xy, speed, lead_t):
    A = path['xy']; s = path['s']
    idx = int(np.argmin(np.sum((A - np.asarray(xy).reshape(1, 2))**2, axis=1)))
    sf = min(float(s[-1]), float(s[idx]) + speed * max(0., lead_t))
    j = int(np.searchsorted(s, sf)); j = max(0, min(j, len(A) - 1))
    return A[j]


def maneuver_from_path(path, xy):
    A = path['xy']; psi = path['psi']; s = path['s']
    j = int(np.argmin(np.sum((A - np.asarray(xy).reshape(1, 2))**2, axis=1)))
    j = max(0, min(j, len(A) - 2))
    dpsi = abs(wrap_pi(float(psi[j + 1]) - float(psi[j])))
    ds = max(float(s[j + 1]) - float(s[j]), 1e-6)
    kappa = dpsi / ds
    return kappa * SPEED * (1.0 + SPEED)


def gamma_for_candidate(cand_xy, auv_xyz_pred, P0_xy, maneuver_m, energy_penalty=0.0):
    J, slant = J_meas_from_geometry(cand_xy, auv_xyz_pred)
    _one_way = slant / C_SOUND
    d_steps = int(round((2.0 * _one_way + PROC +
                         (_one_way if DOWNLINK_LEG else 0.0)) / DT))
    d_steps = max(1, d_steps)
    P_future = P0_xy.copy()
    q_xy = Q_XY_BASE * (1.0 + 0.15 * maneuver_m)
    Qd = d_steps * q_xy * np.eye(2)
    P_future = P_future + Qd
    P_future = 0.5 * (P_future + P_future.T) + 1e-9 * np.eye(2)
    P_post = np.linalg.inv(np.linalg.inv(P_future) + J)
    if TRIGGER_OBJECTIVE == "D":
        gain = (math.log(np.linalg.det(P_future)) -
                math.log(np.linalg.det(P_post) + 1e-12))
    else:
        gain = np.trace(P_future) - np.trace(P_post)
    staleness = math.exp(-STALENESS_LAMBDA * d_steps)
    comm_penalty = COMMUNICATION_LOSS_PROB * COMM_PENALTY_SCALE
    return gain * staleness - comm_penalty, slant


def coverage_penalty(cand_xy, auv_xy_pred):
    r = math.hypot(cand_xy[0] - auv_xy_pred[0], cand_xy[1] - auv_xy_pred[1])
    safe = COVERAGE_MARGIN * OPER_RANGE          # keep AUVs inside the REAL USBL reach (OPER_RANGE), not stale MAXR
    if r <= safe: return 0.0
    over = (r - safe) / (OPER_RANGE - safe)
    return over * over


def combined_coverage(cand_xy, per_h):
    """Coverage cost aggregated over the AUVs at this horizon step. per_h = list of (xyz3, P0g, man, pf) tuples.
    COST_BOWL=0 (default): MAX of the plateau penalty -- flat/zero within COVERAGE_MARGIN*range, so no gradient inside
    the reachable region (the ORIGINAL, wander-prone shape). COST_BOWL=1: MEAN of (dist/OPER_RANGE)^2 -- a BOWL whose
    minimum is the fleet centroid, with a downhill slope EVERYWHERE, so even a short-horizon optimizer feels the pull."""
    if COST_BOWL:
        s = 0.0
        for (_x3, _P, _m, pf) in per_h:
            _r = math.hypot(cand_xy[0] - pf[0], cand_xy[1] - pf[1]) / OPER_RANGE
            s += _r * _r
        return s / max(1, len(per_h))
    cov = 0.0
    for (_x3, _P, _m, pf) in per_h:
        c = coverage_penalty(cand_xy, pf)
        if c > cov: cov = c
    return cov


def _combine_gains(gains):
    """Combine the per-AUV info-gains for POSITIONING per MPPI_MULTI_AUV (max/sum/min)."""
    if not gains:
        return -1e18
    if MPPI_MULTI_AUV == "sum":
        return float(sum(gains))
    if MPPI_MULTI_AUV == "min":
        return float(min(gains))
    return float(max(gains))       # default "max" = original behaviour


def mppi_plan(asv_xy, asv_hdg, AUVS, t, P0s, nom, mrng):
    K, dt = MPPI_K, MPPI_DT
    if AUTO_HORIZON:
        # Non-dimensionalize: horizon derives from the fleet scale so its reach always covers the fleet. fleet_reach
        # = farthest ASV->AUV predicted distance (truth-free: planned path + last fix, never truth).
        _reach = 0.0
        for a in AUVS:
            _st = max(1.0, (t - a.t_last_fix) if a.t_last_fix > -1e8 else 1.0)
            _pf = predict_on_path(a.path, a.last_fix_xy, SPEED, _st)
            _d = math.hypot(_pf[0] - asv_xy[0], _pf[1] - asv_xy[1])
            if _d > _reach: _reach = _d
        H = int(min(MPPI_H_MAX, max(MPPI_H, math.ceil(HORIZON_K * _reach / (V_ASV * dt)))))
        _MPPI_H_LOG.append(H)
        if nom.shape[0] != H:                          # resize the carried nominal to the new horizon length
            _n = np.zeros((H, 2)); _n[:, 1] = ASV_CRUISE
            _m = min(H, nom.shape[0]); _n[:_m] = nom[:_m]
            nom = _n
    else:
        H = MPPI_H
    # METHOD B (predictive pre-positioning): per-AUV URGENCY weight = exp(-t_thresh / TAU_URG), where t_thresh is
    # the predicted time until this AUV's POS-trace crosses REQUEST_BUDGET. Growth uses the leading terms of
    # d/dt trace(P_pos) = 3*Q_XY_BASE/DT (direct Q) + 2*(cov(x,vx)+cov(y,vy)+cov(z,vz)) (the FPFt velocity pump) --
    # truth-free (reads only the AUV's own P). Off -> all weights 1.0 (baseline unchanged).
    if USE_PREDICTIVE_POS:
        urg = []
        _I = EKFOOSM
        for a in AUVS:
            _P = a.shadow.P     # ASV-side arch: predictive weighting reads the SHADOW, never the real EKF
            _tr = float(np.trace(_P[np.ix_(_I.POS_IDX, _I.POS_IDX)]))
            _cross = _P[_I.IX_X, _I.IX_VX] + _P[_I.IX_Y, _I.IX_VY] + _P[_I.IX_Z, _I.IX_VZ]
            _growth = 3.0 * Q_XY_BASE / DT + 2.0 * max(0.0, float(_cross))
            _t_thresh = max(0.0, (ASV_TRIGGER - _tr) / max(_growth, 1e-6))
            urg.append(math.exp(-_t_thresh / TAU_URG))
    else:
        urg = [1.0] * len(AUVS)
    auv_fut = []
    for h in range(H):
        per = []
        for ki, a in enumerate(AUVS):
            stale = max(1.0, (t - a.t_last_fix) if a.t_last_fix > -1e8 else 1.0)
            pf = predict_on_path(a.path, a.last_fix_xy, SPEED, stale + h * dt)
            man = maneuver_from_path(a.path, pf)
            P0g = P0s[ki] + (h * dt / DT) * Q_XY_BASE * np.eye(2)
            per.append((np.array([pf[0], pf[1], a.depth_reported]), P0g, man, pf))
        auv_fut.append(per)
    eps = np.stack([mrng.normal(0.0, MPPI_SIGMA,   size=(K, H)),
                    mrng.normal(0.0, MPPI_SIGMA_V, size=(K, H))], axis=-1)
    U = nom[None] + eps
    U[..., 1] = np.clip(U[..., 1], MPPI_VMIN, MPPI_VMAX)
    _surge = ASV_THRUST_PER_MS * U[..., 1]
    _wlim  = MPPI_WMARGIN * ASV_YAW_PER_N * np.minimum(_surge, ASV_FMAX_THRUSTER - _surge)
    _wlim  = np.clip(_wlim, 0.0, MPPI_WMAX)
    U[..., 0] = np.clip(U[..., 0], -_wlim, _wlim)
    costs = np.empty(K)
    for kk in range(K):
        x, y, hd = asv_xy[0], asv_xy[1], asv_hdg
        c = 0.0
        for h in range(H):
            w_h, v_h = U[kk, h, 0], U[kk, h, 1]
            hd += w_h * dt
            x  += v_h * math.cos(hd) * dt
            y  += v_h * math.sin(hd) * dt
            cand = (x, y)
            gains = []
            for _ki, (xyz3, P0g, man, pf) in enumerate(auv_fut[h]):
                g, _ = gamma_for_candidate(cand, xyz3, P0g, man)
                gains.append(g * urg[_ki])       # METHOD B: urgency-weight the gain (off -> *1.0)
            stage = _combine_gains(gains) - COVERAGE_WEIGHT * combined_coverage(cand, auv_fut[h])
            # cost = -(reward) + turn penalty + ENERGY penalty (thrust ~ v^2): energy discourages needless travel/loops.
            c += -(MPPI_DISCOUNT**h) * stage + MPPI_R_CTRL * (w_h * w_h) + MPPI_R_ENERGY * (v_h * v_h)
        costs[kk] = c
    beta = costs.min()
    w = np.exp(-(1.0 / MPPI_LAMBDA) * (costs - beta)); w /= (w.sum() + 1e-12)
    nom_new = nom + (w[:, None, None] * eps).sum(axis=0)
    nom_new[:, 1] = np.clip(nom_new[:, 1], MPPI_VMIN, MPPI_VMAX)
    _surge_n = ASV_THRUST_PER_MS * nom_new[:, 1]
    _wlim_n  = np.clip(MPPI_WMARGIN * ASV_YAW_PER_N *
                        np.minimum(_surge_n, ASV_FMAX_THRUSTER - _surge_n), 0.0, MPPI_WMAX)
    nom_new[:, 0] = np.clip(nom_new[:, 0], -_wlim_n, _wlim_n)
    traj = np.empty((H, 2)); x, y, hd = asv_xy[0], asv_xy[1], asv_hdg
    gsum = 0.0; csum = 0.0
    for h in range(H):
        hd += nom_new[h, 0] * dt
        x  += nom_new[h, 1] * math.cos(hd) * dt
        y  += nom_new[h, 1] * math.sin(hd) * dt
        traj[h] = (x, y)
        gains = []
        for _ki, (xyz3, P0g, man, pf) in enumerate(auv_fut[h]):
            g, _ = gamma_for_candidate((x, y), xyz3, P0g, man)
            gains.append(g * urg[_ki])           # METHOD B: urgency-weight the gain (off -> *1.0)
        gsum += _combine_gains(gains); csum += combined_coverage((x, y), auv_fut[h])
    v_cmd = float(nom_new[0, 1])
    nom_shift = np.concatenate([nom_new[1:], nom_new[-1:]])
    return traj, nom_shift, gsum / H, csum / H, v_cmd


# ============================================================================
# AUV wrapper
# ============================================================================
class AUV:
    def __init__(self, name, path_xy, depth, x0, y0, yaw0, rng, vx_init=SPEED, noise_mult=1.0):
        self.name = name
        self.depth_reported = depth
        self.path_xy = path_xy
        self.path = wrap_path_for_planner(path_xy)
        self.cum_s = cumulative_path_length(path_xy)
        # ASV-side knowledge
        self.last_fix_xy = np.array([x0, y0])
        self.t_last_fix  = -1e9
        self.P_xy_at_lastfix = None
        self.t_P_xy_at_lastfix = -1e9
        self.unc_reported = EKF_INIT_POS_NOISE**2 * 2.0
        self.t_reported   = 0.0
        self.req_pending  = False
        self.n_fixes      = 0
        self.n_requests   = 0
        # Which of the 3 trigger tiers actually caused each DELIVERED fix (safeguard / request / gamma).
        self.n_fix_by_reason = {"safeguard": 0, "request": 0, "gamma": 0}
        self.n_level_fire = 0      # leveling updates actually applied (see LEVEL_GATE)
        self.n_level_skip = 0      # leveling updates skipped because the quasi-static assumption failed
        # Step-6 safety instrumentation: did the AUV stay within USBL comm range?
        self.max_slant_range = 0.0     # m, max true AUV<->ASV slant range over the run
        self.n_range_loss    = 0       # pings lost specifically because slant range > MAXR
        self.max_ekf_err3d   = 0.0     # m, max true 3D EKF position error
        # AUV self-check report (piggyback): discrepancy samples (t_valid, reported_trace, shadow_trace, gap).
        self.shadow_err_samples = []   # each: (t_valid, reported_tr, shadow_tr, gap=reported-shadow)
        self.n_report_rx        = 0    # reports actually received (rode back on a delivered ping)
        # ASV_TIMING (diagnostic): request->receive delay. t_req_raised is sticky (first unanswered raise);
        # answer_delays collects t_deliver - t_req_raised per served chain; n_req_chains counts distinct chains.
        self.t_req_raised = None
        self.answer_delays = []
        self.n_req_chains  = 0
        self.slots_skipped = 0     # METHOD A (adaptive TDMA): consecutive slots this requester was passed over
        # AUV onboard EKF (with OOSM). Known-start init: launch pose is known (surface GPS fix pre-dive),
        # so the estimate starts at truth with a tight P0 (no injected initial offset).
        # IX_VX/IX_VY are WORLD-frame. The vehicle spawns with BODY surge SPAWN_SURGE at heading yaw0, so the
        # correct world seed is SPAWN_SURGE*[cos(yaw0), sin(yaw0)]. The legacy seed (vx=SPEED=1.0, vy=0) is wrong
        # in magnitude AND direction (|err| = 0.972 m/s at yaw0=39.8deg) and, being unobservable, integrates.
        if EKF_SEED_FIX:
            _vx0 = SPAWN_SURGE * math.cos(yaw0)
            _vy0 = SPAWN_SURGE * math.sin(yaw0)
        else:
            _vx0, _vy0 = vx_init, 0.0        # legacy (buggy) seed
        self.noise_mult = noise_mult    # >1 = this AUV has a noisier IMU (asymmetric fleet)
        self.ekf = EKFOOSM(
            x  = x0, y = y0,
            vx = _vx0, vy = _vy0,
            psi= yaw0,
            z  = depth,                 # known deploy depth
            noise_scale = noise_mult,   # EKF Q scaled to MATCH the noisier IMU -> stays calibrated
        )
        # ASV-SIDE SHADOW EKF (NEW architecture). The ASV owns one per AUV. Initialised IDENTICALLY to the
        # real EKF -- the deploy pose is KNOWN (surface GPS fix pre-dive) and the IMU spec (noise_scale=Q) is
        # public, so this uses NO privileged information. During the run the ASV drives it with the PLAN's
        # expected IMU (see the shadow-propagation block) and collapses it with its own USBL measurements.
        # ANTI-CHEAT: after this line the shadow NEVER reads auv_truths / the real EKF / the real IMU.
        self.shadow = EKFOOSM(x=x0, y=y0, vx=_vx0, vy=_vy0, psi=yaw0, z=depth, noise_scale=noise_mult)
        self.depth_phase = 0.0          # per-AUV phase for the commanded depth profile (set after build)
        # Controller state
        self.psi_cmd     = yaw0
        self.e_int       = 0.0
        self.closest_idx = 0
        self.closest_idx_true = 0   # SCORING ONLY: separate path cursor for TRUE cross-track (never used by control)
        self.est_path_dist = 0.0        # EKF estimate's distance to the planned path (for TRIGGER_ON_PATH_ERR)
        # IMU biases (constant per run). Accel on all 3 axes; gyro bias default 0 (knob for the sweep).
        self.imu_bias_x   = rng.normal(0, IMU_BIAS_INIT)
        self.imu_bias_y   = rng.normal(0, IMU_BIAS_INIT)
        self.imu_bias_z   = rng.normal(0, IMU_BIAS_INIT)
        self.gyro_bias_x  = rng.normal(0, GYRO_BIAS_INIT)
        self.gyro_bias_y  = rng.normal(0, GYRO_BIAS_INIT)
        self.gyro_bias_z  = rng.normal(0, GYRO_BIAS_INIT)
        self.compass_bias = rng.normal(0, COMPASS_BIAS)

    def unc_extrapolated(self, t_now):
        rate = 2 * Q_XY_BASE / DT
        return self.unc_reported + rate * max(0., t_now - self.t_reported)


# ============================================================================
# Build mission: TWO AUVs side-by-side
# ============================================================================
print("[setup] Building two side-by-side Dubins lawnmowers...")

# Enlarge the survey by LENGTHENING THE LEGS ONLY (LEG_LEN). Leg spacing (60/50 m) + turn radius (TURN_R) are FIXED.
# LEG_LEN=100 -> current geometry EXACTLY.

# ---- VERTICAL (top-down) lawnmower for THIS mission: legs run N-S, AUVs START AT THE TOP and sweep DOWN,
#      stepping east by the 60 m swath. Two blocks side-by-side with a fixed 60 m gap (the ASV corridor). ----
def build_vertical_lawnmower(lane_xs, y_top, y_bottom, R, wp_spacing=1.0):
    wps = []; n = len(lane_xs)
    for i in range(n):
        lx = lane_xs[i]
        if i % 2 == 0: y0, y1 = y_top, y_bottom      # first (and every even) leg: TOP -> BOTTOM
        else:          y0, y1 = y_bottom, y_top      # odd leg: BOTTOM -> TOP
        npts = max(2, int(abs(y1 - y0) / wp_spacing) + 1)
        for k in range(npts):
            f = k / (npts - 1); wps.append((lx, y0 + f * (y1 - y0)))
        if i < n - 1:                                # Dubins semicircle turn to the next lane (bulges in Y)
            lx_next = lane_xs[i + 1]; cx = 0.5 * (lx + lx_next); cy = y1
            narc = max(8, int(math.pi * R / wp_spacing))
            for k in range(1, narc + 1):
                theta = -math.pi / 2 + math.pi * (k / narc)
                px = cx + R * math.sin(theta)
                py = (cy - R * math.cos(theta)) if i % 2 == 0 else (cy + R * math.cos(theta))
                wps.append((px, py))
    return np.array(wps)

V_YTOP   = float(_os_flags.environ.get("V_YTOP", "300"))     # leg height: AUVs start at this Y (top), sweep to 0
V_NLANES = int(_os_flags.environ.get("V_NLANES", "4"))        # lanes per block (each LANE_SPACING=60 m apart)
V_GAP    = float(_os_flags.environ.get("V_GAP", "60"))        # gap between the two blocks (ASV corridor)
# Block A: lanes at X = 0, 60, ... ; Block B: A's right edge + V_GAP, then +60 m each.
LANE_XS_A = [LANE_SPACING * k for k in range(V_NLANES)]
_xB0      = LANE_XS_A[-1] + V_GAP
LANE_XS_B = [_xB0 + LANE_SPACING * k for k in range(V_NLANES)]
path_A = build_vertical_lawnmower(LANE_XS_A, V_YTOP, 0.0, TURN_R)
path_B = build_vertical_lawnmower(LANE_XS_B, V_YTOP, 0.0, TURN_R)
print(f"[setup] VERTICAL top-down: block A lanes X={LANE_XS_A}, block B X={LANE_XS_B}, gap={V_GAP:.0f} m, height={V_YTOP:.0f} m")

print(f"        AUV0 path: {len(path_A)} pts, {cumulative_path_length(path_A)[-1]:.1f} m")
print(f"        AUV1 path: {len(path_B)} pts, {cumulative_path_length(path_B)[-1]:.1f} m")

# AUV2 (only when N_AUVS>=3): a 3rd lawnmower SOUTH of the A/B pair, forming a compact triangle. The three must be
# reachable from a central ASV position (within the USBL OPER_RANGE, allowing for DEPTH_REF standoff) yet spread
# enough that the ASV must MOVE between them -> real queueing, not divergence. Footprint set by LEG_LEN + AUV_SEP.
if N_AUVS >= 3:
    # base lanes + EXTRA_LANES more to the SOUTH at -50 spacing (prepended so the sweep still runs south->north)
    LANE_YS_C = [-130.0 - 50.0 * (EXTRA_LANES - k) for k in range(EXTRA_LANES)] + [-130.0, -80.0, -30.0]
    path_C = build_dubins_lawnmower(AUV_SEP / 2.0, AUV_SEP / 2.0 + LEG_LEN, LANE_YS_C, TURN_R)  # AUV_SEP=150 -> [75,175]
    print(f"        AUV2 path: {len(path_C)} pts, {cumulative_path_length(path_C)[-1]:.1f} m")

# Initial poses (start a bit behind path[0], offset sideways)
def start_pose(path_xy):
    START_BACK = 12.0; START_SIDE = -10.0
    tan0 = math.atan2(path_xy[TANGENT_STEP, 1] - path_xy[0, 1],
                      path_xy[TANGENT_STEP, 0] - path_xy[0, 0])
    sx, sy = -math.sin(tan0), math.cos(tan0)
    x = path_xy[0, 0] - START_BACK * math.cos(tan0) + START_SIDE * sx
    y = path_xy[0, 1] - START_BACK * math.sin(tan0) + START_SIDE * sy
    yaw = math.atan2(path_xy[0, 1] - y, path_xy[0, 0] - x)
    return x, y, yaw

ax0, ay0, ayaw0 = start_pose(path_A)
bx0, by0, byaw0 = start_pose(path_B)
if N_AUVS >= 3:
    cx0, cy0, cyaw0 = start_pose(path_C)

# ASV starts at the geometric centroid of the AUVs (2-AUV: midpoint; 3-AUV: 3-way centroid), 40 m south.
if N_AUVS >= 3:
    asv_x0 = (ax0 + bx0 + cx0) / 3.0
    asv_y0 = (ay0 + by0 + cy0) / 3.0 - 40.0
else:
    asv_x0 = 0.5 * (ax0 + bx0)
    asv_y0 = 0.5 * (ay0 + by0) - 40.0

_agents = [
    {"agent_name": "auv0", "agent_type": "TorpedoAUV",
     "location": [ax0, ay0, Z_START],
     "rotation": [0, 0, math.degrees(ayaw0)]},
    {"agent_name": "auv1", "agent_type": "TorpedoAUV",
     "location": [bx0, by0, Z_START],
     "rotation": [0, 0, math.degrees(byaw0)]},
]
if N_AUVS >= 3:
    _agents.append({"agent_name": "auv2", "agent_type": "TorpedoAUV",
                    "location": [cx0, cy0, Z_START],
                    "rotation": [0, 0, math.degrees(cyaw0)]})
_agents.append({"agent_name": "asv0", "agent_type": "SurfaceVessel",
                "location": [asv_x0, asv_y0, 0.0],
                "rotation": [0, 0, 0]})
scenario = {"name": "STAGE3", "ticks_per_sec": int(1.0 / DT), "agents": _agents}
env = mss_env.make(scenario, depth_ref=DEPTH_REF, rpm=RPM_AUV)
env._vehicles["auv0"].vehicle.wn_d = HEADING_WN
env._vehicles["auv1"].vehicle.wn_d = HEADING_WN
if N_AUVS >= 3:
    env._vehicles["auv2"].vehicle.wn_d = HEADING_WN
print(f"[setup] AUV0 at ({ax0:.1f}, {ay0:.1f}), AUV1 at ({bx0:.1f}, {by0:.1f})")
if N_AUVS >= 3:
    print(f"[setup] AUV2 at ({cx0:.1f}, {cy0:.1f})")
print(f"[setup] ASV  at ({asv_x0:.1f}, {asv_y0:.1f})")


# ============================================================================
# Initialise state
# ============================================================================
rng      = np.random.default_rng(RNG_SEED)
mppi_rng = np.random.default_rng(MPPI_SEED)

_nm = lambda nm: ASYM_MULT if nm == ASYM_AUV else 1.0   # per-AUV noise multiplier (asymmetric fleet)
auv0 = AUV("auv0", path_A, Z_START, ax0, ay0, ayaw0, rng, noise_mult=_nm("auv0"))
auv1 = AUV("auv1", path_B, Z_START, bx0, by0, byaw0, rng, noise_mult=_nm("auv1"))
auv0.depth_phase = 0.0          # the AUVs dive/climb out of phase so depth varies independently
auv1.depth_phase = math.pi
AUVS = [auv0, auv1]
# 3rd AUV appended AFTER auv0/auv1 so their rng bias draws are unchanged (2-AUV baseline byte-identical).
if N_AUVS >= 3:
    auv2 = AUV("auv2", path_C, Z_START, cx0, cy0, cyaw0, rng, noise_mult=_nm("auv2"))
    auv2.depth_phase = math.pi / 2
    AUVS.append(auv2)

# Attach a learned-inertial navigator (our TLIO front-end) to each AUV when USE_ML is on.
# Which trained model to use is selected by env ML_CKPT (default: the calibrated sim model).
if USE_ML:
    import sys as _sys, os as _os
    # _ASV_ROOT = ASV_Planner(NoML)/ (located at import time); src/outputs live one level above it (ML/).
    _sys.path.insert(0, _os.path.join(_ASV_ROOT, "..", "src"))
    from ml_navigator import MLNavigator
    _ckpt = _os_flags.environ.get(
        "ML_CKPT",
        _os.path.join(_ASV_ROOT, "..", "outputs", "resnet_sim", "checkpoint_best.pt"),
    )
    for _a in AUVS:
        _a.mlnav = MLNavigator(win_ticks=ML_WIN_TICKS, sim_dt=DT, mock=ML_MOCK,
                               checkpoint=None if ML_MOCK else _ckpt)
        _a.ekf.clone_pose()        # seed the first clone at t0 (known deploy state)
    print(f"[ml] learned front-end ON  (mock={ML_MOCK}, ckpt={'(mock)' if ML_MOCK else _ckpt}, "
          f"window={ML_WIN_TICKS} ticks = {ML_WIN_TICKS*DT:.2f}s)")
else:
    for _a in AUVS:
        _a.mlnav = None

asv_yaw      = 0.0
asv_speed    = 0.0
asv_yaw_rate = 0.0
asv_v_cmd    = ASV_CRUISE
asv_wp       = None
asv_pi       = 0
mppi_nom     = np.zeros((MPPI_H, 2))
mppi_nom[:, 1] = ASV_CRUISE
centroid_ema = None      # CENTROID_HOLD: persistent low-passed centroid target (lazy-init on first slot)

last_slot_t = -SLOT_S
channel_busy_until = -1e9   # half-duplex (MAC_TIMING): ASV cannot transmit again until >= this (reply received)
n_channel_blocked  = 0      # pings the ASV wanted to send but the acoustic channel was still busy
_cycle_samples     = []     # realized interrogation-cycle durations (for the diagnostic)

# Logs per AUV
logs = {a.name: {"true": [], "est": [], "ekf_err": [], "ekf_err_z": [], "ct": [], "ct_true": [],
                 "nees_p": [], "att_err": [],
                 # FIDELITY diagnostic (offline scoring only): the ASV's plan-driven SHADOW cov trace vs the
                 # AUV's REAL EKF cov trace. shadow_tr uses NO truth; ekf_tr is the real EKF (privileged, logged
                 # for scoring only, never fed back into any ASV decision). "t" pairs them in time.
                 "t": [], "shadow_tr": [], "ekf_tr": []} for a in AUVS}
asv_log = []     # [t, asv_x, asv_y] per tick (ASV rides at the surface, z=0)
ping_log = []    # [t, asv_x, asv_y, auv_x, auv_y, auv_z] per emitted USBL ping (for the 3D viz)
log_asv   = []
log_pings = []
log_gamma = []
log_oosm  = []  # per fix delivery: (t_arr, t_valid, latency, dropped_or_applied)
log_asv_wp    = []  # per slot: (t, traj[H,2]) the assigned MPPI waypoints
log_asv_track = []  # per tick: [t, asv_x, asv_y, wp_x, wp_y, asv_yaw, des_hdg, he]

# Pending USBL fixes
pend = []   # (k_idx, t_emit, t_arr, z, R, t_valid)


# ============================================================================
# Main loop
# ============================================================================
print(f"[cfg] USE_ML={USE_ML} ML_Z_ONLY={ML_Z_ONLY} USE_NHC={USE_NHC} USE_COMPASS={USE_COMPASS} "
      f"REQUEST_BUDGET={REQUEST_BUDGET:.0f} SAFEGUARD_S={SAFEGUARD_S:.0f} USE_GAMMA_PING={USE_GAMMA_PING} "
      f"IMU_GRADE={IMU_GRADE:.3g} Q_VEL_FLOOR={Q_VEL_FLOOR:.3g} Q_ATT_FLOOR={Q_ATT_FLOOR:.3g} "
      f"IMU_ACC_NOISE={IMU_ACC_NOISE:.3g} IMU_BIAS_INIT={IMU_BIAS_INIT:.3g} "
      f"GYRO_BIAS_WALK={int(GYRO_BIAS_WALK)} GYRO_BIAS_RW={GYRO_BIAS_RW:.3g} "
      f"LEVEL_INIT_ONLY={int(LEVEL_INIT_ONLY)} GYRO_BIAS_RESET_S={GYRO_BIAS_RESET_S:.3g}")
print(f"\n[run] Running up to {N_TICKS_MAX} ticks ({N_TICKS_MAX*DT:.0f} s)...")
for i in range(N_TICKS_MAX):
    t = i * DT
    st = env.tick()

    # ASV state
    asv_loc = np.asarray(st["asv0"]["LocationSensor"]).ravel()
    asv_xy  = np.array([asv_loc[0], asv_loc[1]])
    asv_log.append([t, float(asv_loc[0]), float(asv_loc[1])])
    # mss_env RotationSensor is already in RADIANS — do NOT re-convert (see AUV note).
    asv_yaw = float(np.asarray(st["asv0"]["RotationSensor"]).ravel()[2])
    asv_vel_world = np.zeros(3)
    if "VelocitySensor" in st["asv0"]:
        vv = np.asarray(st["asv0"]["VelocitySensor"]).ravel()
        asv_speed = float(np.hypot(vv[0], vv[1]))
        asv_vel_world = np.array([vv[0], vv[1], vv[2] if vv.size > 2 else 0.0])   # for Stage-C Doppler
    if "IMUSensor" in st["asv0"]:
        asv_yaw_rate = float(st["asv0"]["IMUSensor"][1][2])

    # AUVs: read state + run onboard stack
    auv_truths = {}
    for a in AUVS:
        s = st[a.name]
        x_true, y_true, z_true = s["LocationSensor"]
        a.vel_world = np.asarray(s.get("VelocitySensor", (0.0, 0.0, 0.0)), float).ravel()   # for Stage-C Doppler
        # mss_env RotationSensor is already RADIANS [roll, pitch, yaw] (NED). The
        # original code applied math.radians() to it — a leftover HoloOcean (degrees)
        # assumption — which scaled yaw by pi/180 and fed the compass a bogus heading
        # on every non-+x lane (psi flipped ~180 deg on the return lanes -> the body
        # accel rotated the wrong way -> runaway). Read it as radians, no conversion.
        rot = np.asarray(s["RotationSensor"]).ravel()
        roll_true, pitch_true, yaw_true = float(rot[0]), float(rot[1]), float(rot[2])
        auv_truths[a.name] = (x_true, y_true, z_true)
        imu = s["IMUSensor"]
        # Realistic sensor model: the accelerometer (specific force, incl. gravity) and gyro (p,q,r) both
        # get a constant bias + white noise. The SAME noisy, biased IMU feeds BOTH the EKF and the network
        # (no pristine feed, no true-attitude gravity removal). Gravity is removed inside the EKF via its
        # ESTIMATED attitude; the network is trained gravity-in, so it is fed the gravity-containing accel.
        # In-run gyro-bias RANDOM WALK: drift the TRUE bias each tick with per-step std GYRO_BIAS_RW
        # (matches the EKF's Q[b_g]=GYRO_BIAS_RW^2 -> truth+model agree -> calibrated). Off by default.
        if GYRO_BIAS_WALK and GYRO_BIAS_RW > 0.0:
            a.gyro_bias_x += rng.normal(0, GYRO_BIAS_RW)
            a.gyro_bias_y += rng.normal(0, GYRO_BIAS_RW)
            a.gyro_bias_z += rng.normal(0, GYRO_BIAS_RW)
        # Periodic gyro-bias RESET (clean recalibration): every GYRO_BIAS_RESET_S s, zero the TRUE bias
        # AND the EKF's gyro-bias estimate/covariance. Bounds heading drift to one interval. Off by default.
        if GYRO_BIAS_RESET_S > 0.0 and int(t / GYRO_BIAS_RESET_S) != int((t - DT) / GYRO_BIAS_RESET_S):
            a.gyro_bias_x = 0.0; a.gyro_bias_y = 0.0; a.gyro_bias_z = 0.0
            a.ekf.reset_gyro_bias()
        accel_meas = np.asarray(imu[0], float) + np.array([a.imu_bias_x, a.imu_bias_y, a.imu_bias_z]) \
                     + rng.normal(0, IMU_ACC_NOISE * a.noise_mult, 3)   # a.noise_mult>1 for the asymmetric AUV
        gyro_meas  = np.asarray(imu[1], float) + np.array([a.gyro_bias_x, a.gyro_bias_y, a.gyro_bias_z]) \
                     + rng.normal(0, GYRO_NOISE_STD * a.noise_mult, 3)

        # EKF predict (records history). Residual accel/gyro biases are estimated by the bias states.
        a.ekf.predict(accel_meas[0], accel_meas[1], accel_meas[2], gyro_meas, DT, t)

        # Accelerometer leveling: gravity reference keeps roll/pitch + accel bias observable (bounds the
        # strapdown now that gravity is removed with the ESTIMATED, not true, attitude).
        if USE_LEVELING:
            if LEVEL_GATE:
                # Apply only where update_leveling's own assumption holds: near-static specific force and low
                # turn rate. Skips the dive and the turns -- exactly the windows that were poisoning attitude.
                _fn = float(np.linalg.norm(accel_meas)); _wn = float(np.linalg.norm(gyro_meas))
                _lev_ok = abs(_fn - GRAVITY) < LEVEL_ACC_TOL and _wn < LEVEL_GYRO_TOL
            else:
                _lev_ok = (not LEVEL_INIT_ONLY) or (t < LEVEL_INIT_S)   # legacy gate
            if _lev_ok:
                a.ekf.update_leveling(accel_meas[0], accel_meas[1], accel_meas[2])
                a.n_level_fire += 1
            else:
                a.n_level_skip += 1

        # ===== ASV-SIDE SHADOW propagation (NEW architecture, truth-free) =====
        # The ASV has NO uplink, so it cannot see this AUV's real IMU. It drives the shadow with the
        # PLAN's EXPECTED IMU for a level, constant-velocity survey:
        #   specific force  ~ [0, 0, -GRAVITY]   (so R@f + [0,0,g] ~ 0 -> zero kinematic accel at level)
        #   body rates      ~ [0, 0, 0]          (turn yaw-rate is 2nd order for the POSITION-cov trace at
        #                                          level: gravity is yaw-invariant, so yaw does not leak into
        #                                          the horizontal position covariance we trigger on)
        # Inputs are ONLY module constants + the shadow's own state -> no auv_truths, no real IMU, no a.ekf.
        _acc_exp  = (0.0, 0.0, -GRAVITY)
        _gyro_exp = np.zeros(3)
        a.shadow.predict(_acc_exp[0], _acc_exp[1], _acc_exp[2], _gyro_exp, DT, t)
        if USE_LEVELING:
            # Mirror the real EKF's accelerometer leveling with the EXPECTED (level) specific force. On the
            # planned level survey the gate is always satisfied (|f|=g, |w|=0), so the shadow's roll/pitch
            # stay bounded exactly as the real EKF's leveling keeps them bounded on the quasi-static legs.
            if LEVEL_GATE:
                _fn_s = float(np.linalg.norm(_acc_exp)); _wn_s = float(np.linalg.norm(_gyro_exp))
                _lev_ok_s = abs(_fn_s - GRAVITY) < LEVEL_ACC_TOL and _wn_s < LEVEL_GYRO_TOL
            else:
                _lev_ok_s = (not LEVEL_INIT_ONLY) or (t < LEVEL_INIT_S)
            if _lev_ok_s:
                a.shadow.update_leveling(_acc_exp[0], _acc_exp[1], _acc_exp[2])

        # --- Learned-displacement update (stochastic cloning), IMU-only, no ground truth ---
        if USE_ML and a.mlnav is not None:
            a.mlnav.push(accel_meas, gyro_meas)           # same noisy IMU the EKF sees
            if not a.ekf.clone_active:
                a.ekf.clone_pose()                        # ensure a clone spans this window
            out = a.mlnav.predict()                       # non-None once a full window accumulated
            if out is not None:
                dp_xyz, Sigma_xyz, _win_dt = out
                a.ekf.update_learned_displacement(dp_xyz, Sigma_xyz, z_only=ML_Z_ONLY)
                a.ekf.clone_pose()                        # re-clone for the next window

        # Compass 1 Hz (disabled for the IMU-only run; see USE_COMPASS)
        if USE_COMPASS and i % 50 == 0:
            psi_meas = yaw_true + a.compass_bias + rng.normal(0, COMPASS_NOISE)
            a.ekf.update_compass(psi_meas)

        # Nonholonomic side-slip constraint (IMU-only heading aid). Applied every tick.
        if USE_NHC:
            a.ekf.update_nonholonomic()
        # DIAGNOSTIC forward-speed aid (default off). Observes the along-track velocity that nothing else does.
        if USE_SPEED_AID:
            if SPEED_AID_SRC == "dvl":
                v_w = np.asarray(s["VelocitySensor"], float)          # true WORLD velocity + noise below
                u_fwd = v_w[0] * math.cos(yaw_true) + v_w[1] * math.sin(yaw_true)   # -> body-forward speed
            else:
                u_fwd = SPEED_AID_CMD                                 # calibrated RPM->speed: NO ground truth
            a.ekf.update_forward_speed(u_fwd + rng.normal(0, SPEED_AID_SIGMA))
        # DIAGNOSTIC depth (pressure) aid (default off). The one absolute reference the IMU cannot provide.
        if USE_DEPTH_AID:
            a.ekf.update_depth(z_true + rng.normal(0, DEPTH_AID_SIGMA))
        # Course-over-ground heading aid (Phase B): psi ~ direction of travel, gated on speed/turn rate.
        if USE_COG:
            a.ekf.update_cog(yaw_rate=float(gyro_meas[2] - a.ekf.x[EKFOOSM.IX_BGZ]))

        # ILOS+FF controller using EKF estimate ONLY
        x_est, y_est, vx_est, vy_est, psi_est = a.ekf.x[:5]
        a.closest_idx, _ = project_to_path(a.path_xy, x_est, y_est, a.closest_idx)
        px, py = a.path_xy[a.closest_idx]
        a.est_path_dist = math.hypot(x_est - px, y_est - py)   # EKF estimate's distance to the planned path
        j = min(a.closest_idx + TANGENT_STEP, len(a.path_xy) - 1)
        pi_local = math.atan2(a.path_xy[j, 1] - py, a.path_xy[j, 0] - px)
        e = -math.sin(pi_local) * (x_est - px) + math.cos(pi_local) * (y_est - py)
        ff_idx = int(np.searchsorted(a.cum_s, a.cum_s[a.closest_idx] + FF_DIST))
        ff_idx = min(ff_idx, len(a.path_xy) - 1)
        ff_j   = min(ff_idx + TANGENT_STEP, len(a.path_xy) - 1)
        pi_ff  = math.atan2(a.path_xy[ff_j, 1] - a.path_xy[ff_idx, 1],
                            a.path_xy[ff_j, 0] - a.path_xy[ff_idx, 0])
        a.e_int += DT * DELTA_LOS * e / (DELTA_LOS**2 + e**2)
        psi_raw  = pi_ff - math.atan2(e + KI_LOS * a.e_int, DELTA_LOS)
        a.psi_cmd = a.psi_cmd + wrap_pi(psi_raw - a.psi_cmd)
        # Commanded depth follows a slow profile so depth is observable (not held constant).
        # DIVE_TIME>0: surface start, ramp depth 0->DEPTH_REF over DIVE_TIME s then hold (sinusoid still available).
        z_hold = DEPTH_REF * min(1.0, t / DIVE_TIME) if DIVE_TIME > 0.0 else DEPTH_REF
        z_cmd = z_hold + DEPTH_AMP * math.sin(2 * math.pi * t / DEPTH_PERIOD + a.depth_phase)
        env.act(a.name, np.array([a.psi_cmd, z_cmd, 0.0, 0.0, 0.0]))
        # Close the INNER autopilot loop on the navigator's estimate, not on ground truth (see AUTOPILOT_ON_EST).
        # Mirrors a real vehicle: attitude/depth from the INS, angular rates straight from the (bias-corrected)
        # rate gyro. env.tick() sits at the top of the loop, so this is consumed next tick -- the same one-tick
        # lag psi_cmd already has via env.act().
        if AUTOPILOT_ON_EST:
            _E = EKFOOSM; _xh = a.ekf.x
            _ph, _th, _ps = _xh[_E.IX_PHI], _xh[_E.IX_THETA], _xh[_E.IX_PSI]
            # DIAGNOSTIC (FEED_TRUE_PITCH): feed the controller the TRUE pitch, everything else estimated -> isolates
            # whether the depth wobble is the noisy est-PITCH. Default off = unchanged (est pitch).
            if FEED_TRUE_PITCH:
                _th = pitch_true
            _eta_hat = np.array([_xh[_E.IX_X], _xh[_E.IX_Y], _xh[_E.IX_Z], _ph, _th, _ps])
            _uvw = _Rzyx(_ph, _th, _ps).T @ np.array([_xh[_E.IX_VX], _xh[_E.IX_VY], _xh[_E.IX_VZ]])
            _pqr = gyro_meas - np.array([_xh[_E.IX_BGX], _xh[_E.IX_BGY], _xh[_E.IX_BGZ]])
            env.set_estimate(a.name, _eta_hat, np.concatenate([_uvw, _pqr]))

        # Log per-AUV
        z_est = a.ekf.x[EKFOOSM.IX_Z]
        if DIVE_TIME > 0.0:
            a.depth_reported = z_est   # AUV reports its own (descending) depth estimate to the ASV for USBL geometry
        err_xy = math.hypot(x_true - x_est, y_true - y_est)
        err_z  = abs(z_true - z_est)
        logs[a.name]["true"].append([t, x_true, y_true, z_true])
        logs[a.name]["est"].append([t, x_est, y_est, z_est])
        logs[a.name]["ekf_err"].append(err_xy)
        logs[a.name]["ekf_err_z"].append(err_z)
        logs[a.name]["ct"].append(abs(e))
        # --- TRUE cross-track: SCORING ONLY (truth used to grade path-following, exactly like ekf_err above;
        #     NEVER fed to control/estimation -- control still uses x_est/y_est). Projects the TRUE position
        #     onto its own path cursor, independent of the controller's closest_idx. ---
        a.closest_idx_true, _ = project_to_path(a.path_xy, x_true, y_true, a.closest_idx_true)
        _pxt, _pyt = a.path_xy[a.closest_idx_true]
        _jt = min(a.closest_idx_true + TANGENT_STEP, len(a.path_xy) - 1)
        _pit = math.atan2(a.path_xy[_jt, 1] - _pyt, a.path_xy[_jt, 0] - _pxt)
        _e_true = -math.sin(_pit) * (x_true - _pxt) + math.cos(_pit) * (y_true - _pyt)
        logs[a.name]["ct_true"].append(abs(_e_true))
        # Consistency: position NEES (3-DOF; consistent E[NEES]=3) + attitude errors (deg).
        e_p = np.array([x_true - x_est, y_true - y_est, z_true - z_est])
        P_pos = a.ekf.P[np.ix_(EKFOOSM.POS_IDX, EKFOOSM.POS_IDX)]
        try:
            nees_p = float(e_p @ np.linalg.solve(P_pos, e_p))
        except np.linalg.LinAlgError:
            nees_p = float("nan")
        logs[a.name]["nees_p"].append(nees_p)
        # FIDELITY: ASV shadow trace (truth-free) vs real EKF trace (privileged, offline scoring only).
        _pos_ix = np.ix_(EKFOOSM.POS_IDX, EKFOOSM.POS_IDX)
        logs[a.name]["t"].append(t)
        logs[a.name]["shadow_tr"].append(float(np.trace(a.shadow.P[_pos_ix])))
        logs[a.name]["ekf_tr"].append(float(np.trace(P_pos)))
        logs[a.name]["att_err"].append([
            math.degrees(abs(wrap_pi(roll_true  - a.ekf.x[EKFOOSM.IX_PHI]))),
            math.degrees(abs(wrap_pi(pitch_true - a.ekf.x[EKFOOSM.IX_THETA]))),
            math.degrees(abs(wrap_pi(yaw_true   - psi_est))),
        ])
        # Safety: true AUV<->ASV slant range (ASV is at the surface, z=0) and 3D EKF error.
        sl_now = math.sqrt((x_true - asv_loc[0])**2 + (y_true - asv_loc[1])**2 + z_true**2)
        a.max_slant_range = max(a.max_slant_range, sl_now)
        a.max_ekf_err3d   = max(a.max_ekf_err3d, math.sqrt(err_xy**2 + err_z**2))

    # Deliver pending USBL fixes (OOSM at t_valid)
    arrived = [p for p in pend if t >= p[2]]
    for p in arrived:
        pend.remove(p)
        k_idx, t_emit, t_arr, z, R, t_valid, fix_reason, report = p
        a = AUVS[k_idx]
        # AUV self-check report rode back on this reply (snapshotted at emission). Record the shadow's error:
        # gap = (AUV's own belief) - (ASV's shadow guess), both at the same emission instant -> no clock sync.
        # DIAGNOSTIC ONLY: never affects triggering. shadow-only pinging is untouched.
        if report is not None:
            reported_tr, shadow_tr_emit = report
            gap = reported_tr - shadow_tr_emit
            a.shadow_err_samples.append((t_valid, reported_tr, shadow_tr_emit, gap))
            a.n_report_rx += 1
        ok = a.ekf.apply_fix_oosm(t_valid, z[0], z[1], z[2], R, DT)
        # Mirror the fix on the ASV's SHADOW with the SAME USBL measurement (z, R) the ASV just made. This
        # collapses the shadow's covariance exactly as the real EKF's, closing the ASV-side trigger loop.
        # Truth-free: z is the ASV's own noisy sensor reading (usbl(asv, auv_true)+noise), NOT the AUV's truth.
        a.shadow.apply_fix_oosm(t_valid, z[0], z[1], z[2], R, DT)
        # Optional SHADOW_ADAPT (default OFF): act only on the TREND. If the running-mean gap over the last N
        # reports shows the shadow persistently UNDER-estimating (mean gap > band), nudge the shadow's post-fix
        # position covariance up so it stops pinging too late. A single sample never acts (needs N).
        if SHADOW_ADAPT and report is not None and len(a.shadow_err_samples) >= SHADOW_ADAPT_N:
            _mean_gap = float(np.mean([s[3] for s in a.shadow_err_samples[-SHADOW_ADAPT_N:]]))
            if _mean_gap > SHADOW_ADAPT_BAND:
                for _i in EKFOOSM.POS_IDX:
                    a.shadow.P[_i, _i] += _mean_gap / len(EKFOOSM.POS_IDX)
                a.shadow.P = 0.5 * (a.shadow.P + a.shadow.P.T)
        if ok:
            a.n_fixes += 1
            if fix_reason in a.n_fix_by_reason:
                a.n_fix_by_reason[fix_reason] += 1   # attribute the DELIVERED fix to its trigger tier
            a.last_fix_xy = z[:2].copy()
            a.t_last_fix  = t_valid               # use t_valid (when fix was taken)
            a.P_xy_at_lastfix = a.shadow.P[:2, :2].copy()   # ASV-side: the SHADOW's post-fix cov (no real EKF)
            a.t_P_xy_at_lastfix = t_valid
            a.unc_reported = float(np.trace(a.P_xy_at_lastfix))
            a.t_reported   = t_valid
            a.req_pending  = False
            if ASV_TIMING and a.t_req_raised is not None:
                # request answered: delay = receive-time (this tick, t >= t_arr) minus first raise
                a.answer_delays.append(t - a.t_req_raised)
                a.t_req_raised = None
            log_pings.append((t_emit, t_arr, z[0], z[1], f"delivered-{a.name}"))
            log_oosm.append((t_arr, t_valid, t_arr - t_valid, "applied"))
        else:
            log_oosm.append((t_arr, t_valid, t_arr - t_valid, "dropped"))

    # ===== ASV-SIDE covariance trigger (NEW architecture: no AUV uplink request) =====
    # The ASV -- NOT the AUV -- decides who needs a fix, using its own per-AUV SHADOW EKF. When the shadow's
    # position-covariance trace crosses ASV_TRIGGER, the ASV flags that AUV as a ping candidate (a.req_pending
    # is now an ASV-side flag, not an acoustic message). This reads the SHADOW ONLY -- never a.ekf.P (the real
    # EKF), never auv_truths. Removes the AUV self-request entirely (an uplink may never arrive).
    if USE_ASV_TRIGGER:
        pos_ix = np.ix_(EKFOOSM.POS_IDX, EKFOOSM.POS_IDX)
        for a in AUVS:
            if not a.req_pending:
                tr_hat = float(np.trace(a.shadow.P[pos_ix]))   # ASV's plan-driven estimate; NO truth
                if tr_hat > ASV_TRIGGER:
                    a.req_pending = True
                    a.n_requests += 1
                    if ASV_TIMING and a.t_req_raised is None:
                        a.t_req_raised = t          # sticky: first unanswered raise of this chain
                        a.n_req_chains += 1
                    if TRACE_DIAG:
                        P = a.shadow.P; I = EKFOOSM
                        print(f"[tracediag-shadow] {a.name} t={t:6.1f} dt_fix={t - a.t_last_fix:5.1f} tr_hat={tr_hat:6.2f} | "
                              f"posVar x={P[I.IX_X,I.IX_X]:.3f} y={P[I.IX_Y,I.IX_Y]:.3f} z={P[I.IX_Z,I.IX_Z]:.3f} | "
                              f"velVar vx={P[I.IX_VX,I.IX_VX]:.4f} vy={P[I.IX_VY,I.IX_VY]:.4f} vz={P[I.IX_VZ,I.IX_VZ]:.4f} | "
                              f"psiVar={P[I.IX_PSI,I.IX_PSI]:.5f}", flush=True)

    # ============================================================
    # ASV SLOT: MPPI + 3-tier ping trigger
    # ============================================================
    if t - last_slot_t >= SLOT_S:
        last_slot_t = t

        # P0s for MPPI positioning -- now the SHADOW's actual 2x2 position covariance (the ASV's real,
        # continuously-propagated covariance model), NOT the old crude linear extrapolation. Truth-free:
        # the shadow reads only the plan + the ASV's own USBL fixes. This is strictly honest and better.
        P0s = []
        for a in AUVS:
            Pa = a.shadow.P[:2, :2].copy()
            Pa = 0.5 * (Pa + Pa.T) + 1e-6 * np.eye(2)
            P0s.append(Pa)

        # MPPI plan  (or the naive centroid baseline)
        if ASV_NAIVE:
            # trivial baseline: steer straight to the AUV CENTROID (truth-free -> uses each AUV's last-known
            # position last_fix_xy, the same knowledge the MPPI planner has). Isolates the value of MPPI positioning.
            _cx = float(np.mean([a.last_fix_xy[0] for a in AUVS]))
            _cy = float(np.mean([a.last_fix_xy[1] for a in AUVS]))
            if CENTROID_HOLD:
                # SETTLE: low-pass the target so it glides (not hops on each fix), and slow to a crawl near it so it
                # settles instead of orbiting. Truth-free (mean of last_fix_xy, same knowledge as the raw centroid).
                if centroid_ema is None:
                    centroid_ema = np.array([_cx, _cy])
                else:
                    centroid_ema = centroid_ema + CENTROID_EMA * (np.array([_cx, _cy]) - centroid_ema)
                _cx, _cy = float(centroid_ema[0]), float(centroid_ema[1])
                _dist = math.hypot(_cx - asv_xy[0], _cy - asv_xy[1])
                asv_v_cmd = float(min(ASV_CRUISE, max(CENTROID_VHOLD, CENTROID_KV * _dist)))
            else:
                asv_v_cmd = ASV_CRUISE
            traj = np.array([[_cx, _cy]]); dgi = dci = 0.0
        else:
            traj, mppi_nom, dgi, dci, asv_v_cmd = mppi_plan(
                asv_xy, asv_yaw, AUVS, t, P0s, mppi_nom, mppi_rng
            )
        asv_wp = traj; asv_pi = 0
        log_gamma.append([t, float(dgi), float(dci), float(asv_v_cmd)])
        # record the full assigned MPPI trajectory for the waypoint-tracking diagnostic
        log_asv_wp.append((t, traj.copy()))

        # 3-tier ping trigger
        ping_target = None; ping_reason = ""
        # Tier 1: SAFEGUARD (most stale)
        ages = [(k, (t - a.t_last_fix) if a.t_last_fix > -1e8 else 1e9)
                for k, a in enumerate(AUVS)]
        stale_ks = [(k, age) for k, age in ages if age > SAFEGUARD_S]
        if stale_ks:
            ping_target = max(stale_ks, key=lambda x: x[1])[0]
            ping_reason = "safeguard"
        # Tier 2: ASV-TRIGGER -- greedy (highest SHADOW trace) OR adaptive TDMA (credit + anti-starvation).
        # Ranking uses the SHADOW's covariance (a.shadow.P), never the real EKF -- the ASV has no uplink.
        if ping_target is None and USE_ASV_TRIGGER:
            req_ks = [k for k, a in enumerate(AUVS) if a.req_pending]
            if req_ks:
                if USE_ADAPTIVE_TDMA:
                    # anti-starvation first: serve the longest-skipped requester if any hit the cap
                    starved = [k for k in req_ks if AUVS[k].slots_skipped >= TDMA_MAX_SKIP]
                    if starved:
                        ping_target = max(starved, key=lambda k: AUVS[k].slots_skipped)
                    else:
                        # credit = need (SHADOW trace) * staleness (time since last fix) -> fairer than trace alone
                        def _cred(k):
                            a = AUVS[k]
                            tr = float(np.trace(a.shadow.P[:2, :2]))
                            stale = max(1.0, (t - a.t_last_fix) if a.t_last_fix > -1e8 else 1.0)
                            return tr * stale
                        ping_target = max(req_ks, key=_cred)
                    ping_reason = "request"
                    for k in req_ks:          # chosen resets; other requesters accrue a skip
                        AUVS[k].slots_skipped = 0 if k == ping_target else AUVS[k].slots_skipped + 1
                else:
                    ping_target = max(req_ks, key=lambda k: float(np.trace(AUVS[k].shadow.P[:2, :2])))
                    ping_reason = "request"
        # Tier 3: Γ-greedy (OPPORTUNISTIC — pings the best-value AUV every slot even when nobody asked).
        # This fills every slot and makes total fixes independent of the covariance trigger, masking the
        # "fewer fixes" win. USE_GAMMA_PING=0 makes the ASV ping ON-DEMAND ONLY (safeguard+request), so a
        # request reduction becomes a real fix reduction — the energy-saving objective of the project.
        if ping_target is None and USE_GAMMA_PING:
            best_g = -1e18; best_k = None
            for k, a in enumerate(AUVS):
                stale = max(1.0, t - a.t_last_fix if a.t_last_fix > -1e8 else 1.0)
                pf = predict_on_path(a.path, a.last_fix_xy, SPEED, stale)
                g, _ = gamma_for_candidate(
                    asv_xy,
                    np.array([pf[0], pf[1], a.depth_reported]),
                    P0s[k],
                    maneuver_from_path(a.path, pf)
                )
                if g > best_g: best_g = g; best_k = k
            if best_g > 0.0:
                ping_target = best_k; ping_reason = "gamma"

        # Emit ping. Half-duplex (MAC_TIMING): the ASV cannot start a new interrogation while its previous
        # exchange is still in flight -> it WAITS for the reply. Event-driven (no true-range prediction): if the
        # channel is still busy this slot, skip the ping (it will re-select when free).
        if ping_target is not None and MAC_TIMING and t < channel_busy_until:
            n_channel_blocked += 1
            ping_target = None
        if ping_target is not None:
            a = AUVS[ping_target]
            x_true, y_true, z_true = auv_truths[a.name]
            auv_true = np.array([x_true, y_true, z_true])
            res = usbl(asv_xy, auv_true, t, rng, asv_vel=asv_vel_world, auv_vel=getattr(a, "vel_world", None))
            if res is not None:
                t_emit, t_arr, z, R, t_valid = res
                if MAC_TIMING:
                    channel_busy_until = t_arr           # busy until the exchange physically completes (reply+dl+guard)
                    _cycle_samples.append(t_arr - t_emit)
                # PIGGYBACK: snapshot, at THIS emission instant (ASV clock), the AUV's own real-EKF trace (its
                # belief, which rides back on the reply) and the ASV's shadow trace. Pairing both at the same
                # instant needs NO clock sync. Reading a.ekf.P here only BUILDS the AUV's outgoing message; it
                # feeds no ASV decision (triggering stays shadow-only), so the anti-cheat rule is intact.
                if USE_AUV_REPORT:
                    _pix = np.ix_(EKFOOSM.POS_IDX, EKFOOSM.POS_IDX)
                    report = (float(np.trace(a.ekf.P[_pix])), float(np.trace(a.shadow.P[_pix])))
                else:
                    report = None
                pend.append((ping_target, t_emit, t_arr, z, R, t_valid, ping_reason, report))
                log_pings.append((t_emit, t_arr, z[0], z[1],
                                  f"emit-{ping_reason}-{a.name}"))
                ping_log.append([t, float(asv_xy[0]), float(asv_xy[1]),
                                 float(x_true), float(y_true), float(z_true)])
            else:
                # Split the loss cause: out-of-range (the safety failure we care about) vs random dropout.
                sl_attempt = math.sqrt((auv_true[0] - asv_xy[0])**2 +
                                       (auv_true[1] - asv_xy[1])**2 + auv_true[2]**2)
                # Classify against the OPERATIONAL range (shallow device rating if set, else old hard MAXR).
                loss_kind = "range" if sl_attempt > OPER_RANGE else "dropout"
                if loss_kind == "range":
                    a.n_range_loss += 1
                log_pings.append((t, t, np.nan, np.nan,
                                  f"lost{loss_kind}-{ping_reason}-{a.name}"))
                if ping_reason == "request":
                    a.req_pending = False

    # ============================================================
    # ASV low-level (every tick)
    # ============================================================
    if asv_wp is not None and len(asv_wp) > 0:
        while asv_pi < len(asv_wp) - 1 and np.hypot(
                asv_wp[asv_pi][0] - asv_xy[0],
                asv_wp[asv_pi][1] - asv_xy[1]) < 3.0:
            asv_pi += 1
        li = asv_pi
        while li < len(asv_wp) - 1 and np.hypot(
                asv_wp[li][0] - asv_xy[0],
                asv_wp[li][1] - asv_xy[1]) < ASV_LOOKAHEAD:
            li += 1
        wp = asv_wp[min(li, len(asv_wp) - 1)]
        des_hdg = math.atan2(wp[1] - asv_xy[1], wp[0] - asv_xy[0])
        he = wrap_pi(des_hdg - asv_yaw)
        slow = max(0.3, 1.0 - abs(he) / (math.pi / 2)) if ASV_TURN_SLOWDOWN else 1.0
        v_tgt = asv_v_cmd * slow
        surge = ASV_THRUST_PER_MS * v_tgt + ASV_KP_V * (v_tgt - asv_speed)
        diff  = ASV_KP_YAW * he - ASV_KD_YAW * asv_yaw_rate
        # SIGN (Otter): yaw moment tau_N = y_pont*(thrust_L - thrust_R) (otter.py:284),
        # so PORT-heavy thrust (f_l > f_r) yields +yaw. To turn toward a positive heading
        # error we need f_l > f_r, i.e. f_l = surge + diff. (The HoloOcean SurfaceVessel
        # used the opposite convention; ported verbatim it steered the Otter the WRONG way
        # -> the ASV drove away from the AUVs. Flipped here for the Otter.)
        f_l = float(np.clip(surge + diff, 0.0, ASV_FMAX_THRUSTER))
        f_r = float(np.clip(surge - diff, 0.0, ASV_FMAX_THRUSTER))
        env.act("asv0", np.array([f_l, f_r]))
        # waypoint-tracking diagnostic: ASV pos, the lookahead target it is chasing,
        # and the heading it commands vs its actual heading.
        log_asv_track.append([t, asv_xy[0], asv_xy[1], float(wp[0]), float(wp[1]),
                              asv_yaw, des_hdg, he])
    else:
        env.act("asv0", np.zeros(2))

    log_asv.append([t, asv_xy[0], asv_xy[1], asv_yaw])

    if i % 5000 == 0:
        msg = f"  t={t:6.1f}s  ASV=({asv_xy[0]:6.1f},{asv_xy[1]:6.1f})"
        for a in AUVS:
            xt, yt, _ = auv_truths[a.name]
            age = (t - a.t_last_fix) if a.t_last_fix > -1e8 else 999.0
            msg += f"  {a.name}=({xt:6.1f},{yt:6.1f}) fixes={a.n_fixes} age={age:5.1f}s"
        print(msg)

    # Stop when both AUVs have traversed
    done = all(a.closest_idx >= len(a.path_xy) - 3 for a in AUVS)
    if done and t > 5.0:
        print(f"[run] Both AUVs traversed paths at t={t:.1f}s")
        break

# ============================================================================
# Stats + Plotting
# ============================================================================
for nm, lg in logs.items():
    lg["true"]      = np.array(lg["true"])
    lg["est"]       = np.array(lg["est"])
    lg["ekf_err"]   = np.array(lg["ekf_err"])
    lg["ekf_err_z"] = np.array(lg["ekf_err_z"])
    lg["ct"]        = np.array(lg["ct"])
    lg["ct_true"]   = np.array(lg["ct_true"])   # SCORING ONLY: true-position cross-track
    lg["nees_p"]    = np.array(lg["nees_p"])
    lg["att_err"]   = np.array(lg["att_err"]) if lg["att_err"] else np.zeros((0, 3))
    lg["t"]         = np.array(lg["t"])
    lg["shadow_tr"] = np.array(lg["shadow_tr"])   # ASV plan-driven cov trace (truth-free)
    lg["ekf_tr"]    = np.array(lg["ekf_tr"])      # AUV real EKF cov trace (offline scoring only)
log_asv   = np.array(log_asv)
log_gamma = np.array(log_gamma) if log_gamma else np.zeros((0, 4))

print(f"\n[result] -------- Per-AUV statistics --------  (ATT_MODE={ATT_MODE}, USE_ML={int(USE_ML)}, "
      f"ML_Z_ONLY={int(ML_Z_ONLY)}, USE_NHC={int(USE_NHC)}, GYRO_BIAS_INIT={GYRO_BIAS_INIT:.4g}, "
      f"IMU_BIAS_INIT={IMU_BIAS_INIT:.4g})")
for a in AUVS:
    lg = logs[a.name]
    ct = lg["ct"]; ee = lg["ekf_err"]; ez = lg["ekf_err_z"]; ctt = lg["ct_true"]
    print(f"  {a.name}: [TRUE ct] mean={ctt.mean():.2f}m p95={np.percentile(ctt, 95):.2f}m max={ctt.max():.2f}m "
          f"(est-ct mean={ct.mean():.2f}m p95={np.percentile(ct, 95):.2f}m)")
    print(f"  {a.name}: ct mean={ct.mean():.2f}m p95={np.percentile(ct, 95):.2f}m  | "
          f"EKF err xy mean={ee.mean():.2f}m max={ee.max():.2f}m  | "
          f"EKF err z mean={ez.mean():.2f}m max={ez.max():.2f}m  | "
          f"fixes={a.n_fixes} reqs={a.n_requests} "
          f"oosm_apply={a.ekf.n_oosm_apply} oosm_drop={a.ekf.n_oosm_skipped}")
    _r = a.n_fix_by_reason
    print(f"        [trigger] delivered fixes by tier: safeguard={_r['safeguard']} "
          f"request={_r['request']} gamma={_r['gamma']}")
    _lt = a.n_level_fire + a.n_level_skip
    if _lt:
        print(f"        [leveling] applied={a.n_level_fire} skipped={a.n_level_skip} "
              f"({100.0*a.n_level_fire/_lt:.1f}% fired)  LEVEL_GATE={int(LEVEL_GATE)}")
    # Consistency: mean position NEES (target ~3, i.e. ANEES~1) + attitude RMS.
    nn = lg["nees_p"]; nn = nn[np.isfinite(nn)]
    anees = (nn.mean() / 3.0) if nn.size else float("nan")
    if lg["att_err"].size:
        r_rms, p_rms, y_rms = np.sqrt((lg["att_err"]**2).mean(axis=0))
    else:
        r_rms = p_rms = y_rms = float("nan")
    print(f"        [consistency] NEES_p mean={nn.mean() if nn.size else float('nan'):.2f} "
          f"(ANEES={anees:.2f}, target 1) | att RMS roll={r_rms:.2f}deg pitch={p_rms:.2f}deg yaw={y_rms:.2f}deg")
    # FIDELITY (the research question): does the ASV's plan-driven SHADOW cov trace track the AUV's REAL EKF
    # trace? A large positive gap = shadow over-estimates -> ASV pings too early (over-fixes); negative =
    # under-estimates -> pings too late (AUV drifts). This is the whole point of the uplink-free architecture.
    st_ = lg["shadow_tr"]; et_ = lg["ekf_tr"]
    if st_.size:
        _gap = st_ - et_
        _rms = float(np.sqrt((_gap**2).mean()))
        print(f"        [fidelity] shadow_tr vs EKF_tr (m^2): RMS gap={_rms:.2f}  mean(shadow-EKF)={_gap.mean():+.2f} "
              f"| shadow mean={st_.mean():.2f} max={st_.max():.2f}  EKF mean={et_.mean():.2f} max={et_.max():.2f}")
    # IN-FIELD self-check (piggyback report): the AUV's OWN belief vs the ASV's shadow, sampled AT FIX TIMES only.
    # gap = reported(AUV) - shadow(ASV). gap<0 => shadow over-estimates (pings early); gap>0 => shadow under-
    # estimates (pings late). This is the deployable stand-in for the offline [fidelity] above (which needs truth-
    # adjacent EKF logging); the two should agree in magnitude -> confirms the in-field monitor is trustworthy.
    if USE_AUV_REPORT and a.shadow_err_samples:
        _rep = np.array([s[1] for s in a.shadow_err_samples])
        _shd = np.array([s[2] for s in a.shadow_err_samples])
        _g   = np.array([s[3] for s in a.shadow_err_samples])
        print(f"        [report] in-field self-check: reports={a.n_report_rx} | reported mean={_rep.mean():.2f} "
              f"shadow mean={_shd.mean():.2f} | gap(rep-shadow) mean={_g.mean():+.2f} RMS={np.sqrt((_g**2).mean()):.2f} m^2"
              + (f" | SHADOW_ADAPT on" if SHADOW_ADAPT else ""))
    # Safety verdict. PHYSICS_OFF: the hard MAXR IS the range limit, so keep the max_slant<MAXR test. Physics mode:
    # range is soft (emerges from SNR, reaches ~km), so MAXR=150 is only a nominal reference -- the physical
    # indicator is whether pings were actually LOST (n_range_loss). Judge on that, not the 150 m line.
    if PHYSICS_OFF:
        safe = "SAFE" if (a.max_slant_range < MAXR and a.n_range_loss == 0) else "OUT-OF-RANGE"
    else:
        safe = "SAFE" if a.n_range_loss == 0 else "COMMS-LOSS"
    _rng_ref = SHALLOW_MAXR if math.isfinite(SHALLOW_MAXR) else MAXR
    _rng_lbl = f"SHALLOW_MAXR={SHALLOW_MAXR:.0f}" if math.isfinite(SHALLOW_MAXR) else f"MAXR={MAXR:.0f}"
    print(f"        [safety] max slant range={a.max_slant_range:.1f}m ({_rng_lbl}) "
          f"range_losses={a.n_range_loss} max_3d_err={a.max_ekf_err3d:.2f}m -> {safe}")
    # ASV_TIMING: request->receive delay (the "answered on time?" metric). A still-open t_req_raised at sim
    # end is one unserved request. served+unserved MUST equal n_req_chains (distinct chains) or accounting is off.
    if ASV_TIMING:
        d = np.array(a.answer_delays, dtype=float)
        served   = d.size
        unserved = 1 if a.t_req_raised is not None else 0
        open_age = (t - a.t_req_raised) if a.t_req_raised is not None else 0.0
        chain_ok = "OK" if (served + unserved == a.n_req_chains) else "MISMATCH"
        if served:
            print(f"        [asv-timing] answer delay: mean={d.mean():.2f} p50={np.percentile(d,50):.2f} "
                  f"p95={np.percentile(d,95):.2f} max={d.max():.2f} s over {served} served chains; "
                  f"unserved={unserved} (open {open_age:.1f}s) | chains={a.n_req_chains} raises={a.n_requests} [{chain_ok}]")
        else:
            print(f"        [asv-timing] no served request chains; unserved={unserved} (open {open_age:.1f}s) "
                  f"| chains={a.n_req_chains} raises={a.n_requests} [{chain_ok}]")

n_emit = sum(1 for p in log_pings if str(p[4]).startswith("emit"))
n_deliv= sum(1 for p in log_pings if str(p[4]).startswith("delivered"))
n_lost = sum(1 for p in log_pings if str(p[4]).startswith("lost"))
print(f"[result] USBL: emitted={n_emit} delivered={n_deliv} lost={n_lost}")

# ASV effort diagnostic: how far / how energetically did the ASV drive? (path length is the clean energy proxy;
# the MPPI energy penalty aims to shrink it without losing fixes/coverage.)
if getattr(log_asv, "ndim", 0) == 2 and log_asv.shape[0] > 1:
    _axy = log_asv[:, 1:3]
    _pathlen = float(np.sum(np.linalg.norm(np.diff(_axy, axis=0), axis=1)))
    _dur = float(log_asv[-1, 0] - log_asv[0, 0])
    print(f"[result] ASV effort: path_length={_pathlen:.0f}m | mean_speed={_pathlen/max(_dur,1e-6):.2f}m/s "
          f"| MPPI_R_ENERGY={MPPI_R_ENERGY:g} MPPI_R_CTRL={MPPI_R_CTRL:g} ASV_NAIVE={int(ASV_NAIVE)}")
    if AUTO_HORIZON and _MPPI_H_LOG:
        _Ha = np.array(_MPPI_H_LOG)
        _jit = float(np.abs(np.diff(_Ha)).mean()) if _Ha.size > 1 else 0.0
        _pin = 100.0 * np.mean((_Ha >= MPPI_H_MAX) | (_Ha <= MPPI_H)) if _Ha.size else 0.0
        print(f"[result] AUTO_HORIZON on: HORIZON_K={HORIZON_K:g} cap={MPPI_H_MAX} combine={MPPI_MULTI_AUV} | derived H "
              f"min={_Ha.min()} med={int(np.median(_Ha))} max={_Ha.max()} (fixed baseline was {MPPI_H}) | "
              f"jitter(mean|dH|)={_jit:.2f} steps/replan | %pinned-at-a-fence={_pin:.0f}%")

# MAC / half-duplex channel diagnostic: is the ASV channel-bound? (only meaningful with MAC_TIMING=1)
if MAC_TIMING:
    _tot_fix = sum(a.n_fixes for a in AUVS)
    _busy = float(np.sum(_cycle_samples)); _util = 100.0 * _busy / max(t, 1e-6)
    _cyc = np.array(_cycle_samples) if _cycle_samples else np.array([0.0])
    _med = float(np.median(_cyc))
    print(f"[result] MAC: MODEM_BPS={MODEM_BPS:.0f} T_REPLY={NBITS/MODEM_BPS:.3f}s | cycle: min={_cyc.min():.2f}s "
          f"med={_med:.2f}s max={_cyc.max():.2f}s | fix_rate={_tot_fix/max(t,1e-6):.2f} Hz (fleet) "
          f"| channel_util={_util:.0f}% | pings_blocked(busy)={n_channel_blocked}")
    if SLOT_S < _med:
        print(f"[result] MAC: WARNING SLOT_S={SLOT_S:.2f}s < median cycle {_med:.2f}s -> decision cadence is below the "
              f"channel cycle; the half-duplex gate throttles the rate to ~1/cycle (extra decisions are blocked).")

colors = {"auv0": "blue", "auv1": "darkorange", "auv2": "green"}
# Per-run output: when RUN_TAG is set, every file for this run lands in Results/<RUN_TAG>/ so runs never mix.
# No RUN_TAG -> flat Results/ (back-compat). Inside the subfolder the filenames drop the tag (folder disambiguates).
_run_tag = _os_flags.environ.get("RUN_TAG", "").strip()
_tag_sfx = ("_" + _run_tag.replace("/", "_")) if _run_tag else ""   # slash-safe: nested RUN_TAG -> valid filename
# Output root pinned to THIS script's folder ("New mission plan/Result/") so results land there whatever the CWD.
_outroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Result")
_outdir  = os.path.join(_outroot, _run_tag) if _run_tag else _outroot
os.makedirs(_outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# FIGURE 1: 2D top-down trajectory ONLY (its own file, no diagnostics).
# ---------------------------------------------------------------------------
fig2d, ax0 = plt.subplots(figsize=(9, 9))
for a in AUVS:
    lg = logs[a.name]; col = colors[a.name]
    ax0.plot(a.path_xy[:, 0], a.path_xy[:, 1], '--', color=col, lw=0.8, alpha=0.5,
             label=f'{a.name} reference')
    ax0.plot(lg["true"][:, 1], lg["true"][:, 2], '-', color=col, lw=1.3,
             label=f'{a.name} truth')
    ax0.plot(lg["est"][:, 1], lg["est"][:, 2], '-', color=col, lw=0.5, alpha=0.4,
             label=f'{a.name} EKF est')
    ax0.plot(lg["true"][0, 1], lg["true"][0, 2], 'o', color=col, ms=10)
    ax0.plot(lg["true"][-1, 1], lg["true"][-1, 2], '*', color=col, ms=14)
ax0.plot(log_asv[:, 1], log_asv[:, 2], 'k-', lw=1.5, label='ASV (Otter)')
ax0.plot(log_asv[0, 1], log_asv[0, 2], 'ks', ms=10, label='ASV start')
ax0.plot(log_asv[-1, 1], log_asv[-1, 2], 'k*', ms=14, label='ASV end')
for p in log_pings:
    if str(p[4]).startswith("delivered") and not np.isnan(p[2]):
        ax0.plot(p[2], p[3], 'm+', ms=6, alpha=0.5)
ax0.set_xlabel('X [m]'); ax0.set_ylabel('Y [m]')
ax0.set_aspect('equal'); ax0.grid(); ax0.legend(loc='best', fontsize=8)
ax0.set_title(f"Trajectory (top-down): 2 AUVs + ASV\npings: emit={n_emit}, deliv={n_deliv}, lost={n_lost}")
fig2d.tight_layout()
out_2d = f"{_outdir}/traj2d{_tag_sfx}.png"
fig2d.savefig(out_2d, dpi=120, bbox_inches='tight'); plt.close(fig2d)
print(f"[plot] Saved {out_2d}")

# ---------------------------------------------------------------------------
# FIGURE 2: 3D trajectory (x, y, z=depth) of both AUVs + the surface ASV.
# Reuses view_traj_3d.py's projection approach; built inline from in-memory logs.
# ---------------------------------------------------------------------------
fig3d = plt.figure(figsize=(11, 9))
ax3d = fig3d.add_subplot(111, projection="3d")
for a in AUVS:
    lg = logs[a.name]; col = colors[a.name]
    tru, est = lg["true"], lg["est"]
    ax3d.plot(tru[:, 1], tru[:, 2], tru[:, 3], '-', color=col, lw=1.3, label=f'{a.name} truth')
    ax3d.plot(est[:, 1], est[:, 2], est[:, 3], '-', color=col, lw=0.5, alpha=0.4, label=f'{a.name} EKF est')
    ax3d.scatter(tru[0, 1], tru[0, 2], tru[0, 3], color=col, marker='o', s=40)
    ax3d.scatter(tru[-1, 1], tru[-1, 2], tru[-1, 3], color=col, marker='*', s=90)
# ASV rides the surface (z = 0)
ax3d.plot(log_asv[:, 1], log_asv[:, 2], np.zeros(len(log_asv)), 'k-', lw=1.5, label='ASV (surface)')
# Delivered fixes: use the ping_log AUV position [.., auv_x, auv_y, auv_z]
if ping_log:
    pg = np.array(ping_log)
    ax3d.scatter(pg[:, 3], pg[:, 4], pg[:, 5], color='red', marker='x', s=15, alpha=0.5, label='pings (AUV pos)')
ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z / depth [m]')
ax3d.set_title('3D trajectory: AUVs (with depth) + ASV on surface')
ax3d.legend(loc='best', fontsize=8)
fig3d.tight_layout()
out_3d = f"{_outdir}/traj3d{_tag_sfx}.png"
fig3d.savefig(out_3d, dpi=120, bbox_inches='tight'); plt.close(fig3d)
print(f"[plot] Saved {out_3d}")

# ---------------------------------------------------------------------------
# FIGURE 3: diagnostics ONLY (no trajectory) -- cross-track, EKF error, OOSM
# acoustic latency, MPPI Gamma/coverage, commanded ASV speed, ASV<->AUV range.
# ---------------------------------------------------------------------------
figd = plt.figure(figsize=(16, 9))
gs = figd.add_gridspec(2, 3)

ax1 = figd.add_subplot(gs[0, 0])
for a in AUVS:
    lg = logs[a.name]
    ax1.plot(lg["true"][:, 0], lg["ct"], '-', color=colors[a.name], lw=0.9,
             label=f'{a.name} (mean={lg["ct"].mean():.1f}m)')
ax1.set_xlabel('time [s]'); ax1.set_ylabel('|cross-track| [m]')
ax1.set_title('AUV path-following error'); ax1.grid(); ax1.legend()

ax2 = figd.add_subplot(gs[1, 0])
for a in AUVS:
    lg = logs[a.name]
    ax2.plot(lg["true"][:, 0], lg["ekf_err"], '-', color=colors[a.name], lw=0.9,
             label=f'{a.name} (mean={lg["ekf_err"].mean():.1f}m)')
ax2.set_xlabel('time [s]'); ax2.set_ylabel('|truth − EKF| [m]')
ax2.set_title('AUV EKF position error'); ax2.grid(); ax2.legend()

ax3 = figd.add_subplot(gs[0, 1])
if log_oosm:
    lat = [r[2] for r in log_oosm if r[3] == "applied"]
    if lat:
        ax3.hist(lat, bins=30, color='c', alpha=0.7)
        ax3.set_xlabel('OOSM acoustic latency (t_arr − t_valid) [s]')
        ax3.set_ylabel('# fixes')
        ax3.set_title(f'OOSM acoustic latency (applied: {len(lat)}, dropped: '
                      f'{sum(1 for r in log_oosm if r[3] == "dropped")})')
        ax3.grid()

ax4 = figd.add_subplot(gs[1, 1])
if len(log_gamma) > 0:
    ax4.plot(log_gamma[:, 0], log_gamma[:, 1], 'b-', label='Γ mean')
    ax4.plot(log_gamma[:, 0], log_gamma[:, 2], 'r-', label='coverage mean')
ax4.set_xlabel('time [s]'); ax4.set_ylabel('value')
ax4.set_title('MPPI diagnostics (Γ, coverage)'); ax4.grid(); ax4.legend()

ax5 = figd.add_subplot(gs[0, 2])
if len(log_gamma) > 0:
    ax5.plot(log_gamma[:, 0], log_gamma[:, 3], 'k-')
ax5.axhline(MPPI_VMIN, color='r', ls=':', label=f'VMIN={MPPI_VMIN}')
ax5.axhline(MPPI_VMAX, color='g', ls=':', label=f'VMAX={MPPI_VMAX}')
ax5.set_xlabel('time [s]'); ax5.set_ylabel('v_cmd [m/s]')
ax5.set_title('MPPI commanded ASV speed'); ax5.grid(); ax5.legend()

ax6 = figd.add_subplot(gs[1, 2])
for a in AUVS:
    lg = logs[a.name]
    d_auv_asv = np.hypot(lg["true"][:, 1] - log_asv[:, 1],
                         lg["true"][:, 2] - log_asv[:, 2])
    ax6.plot(lg["true"][:, 0], d_auv_asv, color=colors[a.name], lw=0.9,
             label=f'ASV↔{a.name}')
ax6.axhline(OPER_RANGE, color='r', ls='--', label=f'USBL range={OPER_RANGE:.0f}m')
ax6.axhline(COVERAGE_MARGIN * OPER_RANGE, color='orange', ls='--',
            label=f'cov margin={COVERAGE_MARGIN * OPER_RANGE:.0f}m')
ax6.set_xlabel('time [s]'); ax6.set_ylabel('ASV→AUV [m]')
ax6.set_title(f'ASV-AUV ranges (must stay < {OPER_RANGE:.0f}m for fixes)'); ax6.grid(); ax6.legend()

figd.tight_layout()
out_diag = f"{_outdir}/diagnostics{_tag_sfx}.png"
figd.savefig(out_diag, dpi=120, bbox_inches='tight'); plt.close(figd)
print(f"[plot] Saved {out_diag}")

# ---------------------------------------------------------------------------
# FIGURE (ASV-side trigger): FIDELITY -- the ASV's plan-driven SHADOW cov trace vs the AUV's REAL EKF
# cov trace. This is the headline diagnostic of the uplink-free architecture: if the two tracks agree,
# the ASV's trigger fires at the right time WITHOUT ever hearing from the AUV. The trigger threshold
# ASV_TRIGGER is drawn so it's clear when each AUV crosses it. (EKF trace = privileged, offline only.)
figf, axf = plt.subplots(len(AUVS), 1, figsize=(10, 3.0 * len(AUVS)), squeeze=False)
for i, a in enumerate(AUVS):
    lg = logs[a.name]; ax = axf[i, 0]
    if lg["shadow_tr"].size:
        ax.plot(lg["t"], lg["shadow_tr"], '-', color='tab:red', lw=1.1,
                label='ASV shadow trace (plan-driven, truth-free)')
        ax.plot(lg["t"], lg["ekf_tr"], '-', color='tab:blue', lw=1.1, alpha=0.85,
                label='AUV real EKF trace (offline scoring)')
        ax.axhline(ASV_TRIGGER, color='k', ls='--', lw=0.9, label=f'ASV_TRIGGER={ASV_TRIGGER:g}')
        # IN-FIELD self-check samples (piggyback): the AUV's reported belief at each fix, which is what the ASV
        # actually gets to see (no truth). These should sit on top of the blue EKF-trace line -> the deployable
        # monitor matches the offline fidelity.
        if USE_AUV_REPORT and a.shadow_err_samples:
            _ts  = [s[0] for s in a.shadow_err_samples]
            _rps = [s[1] for s in a.shadow_err_samples]
            ax.scatter(_ts, _rps, s=18, c='tab:green', zorder=5, marker='o',
                       label='AUV reported belief @ fix (piggyback)')
        _rms = float(np.sqrt(((lg["shadow_tr"] - lg["ekf_tr"])**2).mean()))
        ax.set_title(f'{a.name}: shadow-vs-EKF cov trace  (RMS gap={_rms:.2f} m$^2$, fixes={a.n_fixes})')
    ax.set_xlabel('time [s]'); ax.set_ylabel('trace(P_pos) [m$^2$]'); ax.grid(); ax.legend(fontsize=8)
figf.tight_layout()
out_fid = f"{_outdir}/fidelity{_tag_sfx}.png"
figf.savefig(out_fid, dpi=120, bbox_inches='tight'); plt.close(figf)
print(f"[plot] Saved {out_fid}")

# ---------------------------------------------------------------------------
# FIGURE 4 (ASV_TIMING only): request->receive delay histogram per AUV.
# ---------------------------------------------------------------------------
if ASV_TIMING:
    figt2, axt2 = plt.subplots(figsize=(9, 5))
    any_d = False
    for a in AUVS:
        d = np.array(a.answer_delays, dtype=float)
        if d.size:
            any_d = True
            axt2.hist(d, bins=25, color=colors[a.name], alpha=0.5,
                      label=f'{a.name} (mean={d.mean():.2f}s, n={d.size})')
    axt2.set_xlabel('request→receive delay [s]  (first unanswered raise → fix received)')
    axt2.set_ylabel('# request chains')
    axt2.set_title('ASV answer delay (queueing + contention + acoustic + retries)')
    if any_d:
        axt2.legend()
    axt2.grid()
    figt2.tight_layout()
    out_timing = f"{_outdir}/asv_timing{_tag_sfx}.png"
    figt2.savefig(out_timing, dpi=120, bbox_inches='tight'); plt.close(figt2)
    print(f"[plot] Saved {out_timing}")

# Compact per-run 3D trajectory dump (true + estimated x,y,z) for the interactive viewer.
# ~1-2 MB compressed; only when RUN_TAG is set. Columns of each array: [t, x, y, z].
if _run_tag:
    _traj_path = f"{_outdir}/traj_{_run_tag.replace('/', '_')}.npz"
    _dump = {}
    for a in AUVS:
        _dump[f"{a.name}_true"] = logs[a.name]["true"]
        _dump[f"{a.name}_est"]  = logs[a.name]["est"]
        _dump[f"{a.name}_ct_true"] = logs[a.name]["ct_true"]   # SCORING ONLY: true cross-track series
        _dump[f"{a.name}_ct_est"]  = logs[a.name]["ct"]        # estimate cross-track series (fix-jump artifact)
        _dump[f"{a.name}_path"] = a.path_xy                    # planned lawnmower path (for lane overlay)
    _dump["asv"]   = np.array(asv_log)    if asv_log  else np.zeros((0, 3))
    _dump["pings"] = np.array(ping_log)   if ping_log else np.zeros((0, 6))
    np.savez_compressed(_traj_path, **_dump)
    print(f"[traj] Saved {_traj_path}  (open with: python view_traj_3d.py {_run_tag})")

# ============================================================================
# ASV WAYPOINT-TRACKING DIAGNOSTIC — does the ASV pass through its assigned
# MPPI waypoints?  Left: ASV path overlaid with every assigned waypoint.
# Right-top: distance from ASV to the lookahead target it is chasing (should
# stay small / dip toward 0 as it reaches waypoints).  Right-bottom: commanded
# vs actual heading (should overlap once steering tracks).
# ============================================================================
trk = np.array(log_asv_track) if log_asv_track else np.zeros((0, 8))
figt = plt.figure(figsize=(18, 8))
gst  = figt.add_gridspec(2, 2)

axA = figt.add_subplot(gst[:, 0])
# MPPI is RECEDING-HORIZON: it replans every slot and the ASV executes only the
# first step before the rest of that horizon is discarded. So the executed path is
# expected to track the COMMITTED reference (the 1st waypoint of each slot, orange),
# NOT the faint full horizons (gold), most of which are superseded at the next replan.
for (_, traj) in log_asv_wp:
    axA.plot(traj[:, 0], traj[:, 1], '.', color='gold', ms=1.5, alpha=0.12)
committed = None
if log_asv_wp:
    committed = np.array([[tr[0, 0], tr[0, 1]] for (_, tr) in log_asv_wp])
    axA.plot(committed[:, 0], committed[:, 1], '-o', color='darkorange', lw=1.3,
             ms=3, alpha=0.9, label='committed reference (1st waypoint / slot)')
    axA.plot([], [], '.', color='gold', label='full MPPI horizons (mostly superseded)')
axA.plot(log_asv[:, 1], log_asv[:, 2], 'k-', lw=1.6, label='ASV actual path')
axA.plot(log_asv[0, 1], log_asv[0, 2], 'ks', ms=10, label='ASV start')
for a in AUVS:
    lg = logs[a.name]
    axA.plot(lg["true"][:, 1], lg["true"][:, 2], '-', color=colors[a.name],
             lw=1.0, alpha=0.6, label=f'{a.name} truth')
axA.set_xlabel('X [m]'); axA.set_ylabel('Y [m]'); axA.set_aspect('equal')
axA.grid(); axA.legend(loc='best', fontsize=8)
axA.set_title('ASV actual path vs committed MPPI reference')

axB = figt.add_subplot(gst[0, 1])
if len(trk):
    track_err = np.hypot(trk[:, 1] - trk[:, 3], trk[:, 2] - trk[:, 4])
    axB.plot(trk[:, 0], track_err, 'b-', lw=0.8)
    axB.set_title(f'distance ASV -> lookahead target  '
                  f'(mean={track_err.mean():.1f} m, '
                  f'within {ASV_LOOKAHEAD:.0f} m lookahead)')
axB.axhline(ASV_LOOKAHEAD, color='g', ls=':', label=f'lookahead={ASV_LOOKAHEAD:.0f} m')
axB.set_xlabel('time [s]'); axB.set_ylabel('dist to target [m]')
axB.grid(); axB.legend()

axC = figt.add_subplot(gst[1, 1])
if len(trk):
    axC.plot(trk[:, 0], np.degrees(trk[:, 6]), 'g-', lw=0.7, label='commanded heading')
    axC.plot(trk[:, 0], np.degrees(trk[:, 5]), 'k-', lw=0.7, alpha=0.7, label='actual yaw')
    axC.plot(trk[:, 0], np.degrees(np.abs(trk[:, 7])), 'r-', lw=0.6, alpha=0.6,
             label='|heading error|')
axC.set_xlabel('time [s]'); axC.set_ylabel('heading [deg]')
axC.grid(); axC.legend(fontsize=8)
axC.set_title('commanded vs actual ASV heading')

figt.tight_layout()
out_trk = f"{_outdir}/asv_track{_tag_sfx}.png"
figt.savefig(out_trk, dpi=120, bbox_inches='tight')
print(f"[plot] Saved {out_trk}")
if len(trk):
    te = np.hypot(trk[:, 1] - trk[:, 3], trk[:, 2] - trk[:, 4])
    print(f"[asv-track] dist to lookahead target: mean={te.mean():.1f} m  "
          f"p95={np.percentile(te, 95):.1f} m  max={te.max():.1f} m  "
          f"| |heading err| mean={np.degrees(np.abs(trk[:, 7])).mean():.1f} deg")
# faithful-following: how far the EXECUTED path strays from the committed reference
# (nearest committed-reference vertex to each executed point). This is the honest
# "does it follow the plan" number — small = the ASV hugs the committed waypoints.
if committed is not None and len(committed) > 1 and len(log_asv) > 1:
    P = log_asv[::5, 1:3]                       # decimate executed path
    d_commit = np.sqrt(((P[:, None, :] - committed[None, :, :]) ** 2).sum(-1)).min(1)
    print(f"[asv-track] executed-path deviation from committed reference: "
          f"mean={d_commit.mean():.1f} m  p95={np.percentile(d_commit, 95):.1f} m  "
          f"max={d_commit.max():.1f} m")

ok = all(logs[a.name]["ct"].mean() < 5.0 for a in AUVS)
if ok and n_deliv > 20:
    print("\n[verdict] PASS — both AUVs follow paths, ASV pings both, OOSM working.")
else:
    print("\n[verdict] WARN — check stats above.")
