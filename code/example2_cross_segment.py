# %% [markdown]
# # Worked example 2: a seabed-to-satellite relay across the sea surface
#
# Companion code of *Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey* (M. Jouhari, H. Benaddi, M. Guizani), **Section V-D**. Repository: https://github.com/mohammed-jouhari/PIML-SAGSIN-Survey
#
# **The question.** A sensor on the seabed sends its data to a satellite through a buoy at the surface. Which of two buoys should relay the data, and at which rate, in conditions that were never measured?
#
# **Two physics, one decision.** Sound carries the data under water (acoustic hop), and radio carries it to space (satellite hop). The best buoy depends on both hops, so the decision crosses two segments of SAGSIN.
#
# **What is compared.** A neural network that only sees data, the physical law of each segment calibrated on a few measurements, and these laws completed at the buoy either by a simple formula or by a small neural network.
#
# **Run time** on a free Colab CPU: `MODE = "full"` (5 seeds) takes about 12 to 18 minutes and `MODE = "quick"` (1 seed) about 4 minutes. Only NumPy, SciPy, scikit-learn, and Matplotlib are used. From a terminal, `NB_MODE=quick python example2_cross_segment.py` runs the quick mode.

# %% [markdown]
# ## 1. Scenario
#
# Radio waves do not cross the sea surface (Section III-G of the article), so the sensor sends sound to a buoy, and the buoy relays the data to a LEO satellite by radio. Two buoys can serve as gateway.
#
# | Gateway | Acoustic hop (sensor to buoy) | Satellite hop (buoy to satellite) |
# |---|---|---|
# | **G1**, small drifting buoy | slower when the buoy is far | small antenna pointing up: weak when the satellite is low, and the buoy tilts with the waves |
# | **G2**, larger buoy | slower when the buoy is far | stabilized antenna with more gain |
#
# The wind acts on both hops: it raises the noise of breaking waves at the acoustic receiver, and it makes G1 tilt.
#
# Measurements exist only for buoys within 3 km of the sensor, elevations above 40°, and winds below 8 m/s. The models are tested in three kinds of new conditions.
#
# | Test | Distance of the buoys | Satellite elevation | Wind |
# |---|---|---|---|
# | New geometry | up to 6 km | down to 10° | 2-8 m/s |
# | Storm | up to 3 km | above 40° | 8-14 m/s |
# | Both shifts | 3-6 km | 10-40° | 8-14 m/s |
#
# Three gateway behaviors are simulated, and no model is told which one holds.
#
# - **Full duplex**: the end-to-end rate is the minimum of the two hop rates.
# - **Ideal half duplex (time sharing)**: with the best time split, the rate is $R_aR_s/(R_a+R_s)$.
# - **Practical half duplex**: each packet of $L = 2.048$ kbit also costs a switching delay (Eq. (16) of the article):
# $$R = \frac{L}{L/R_a + L/R_s + T_{\mathrm{sw}}(g, w)}, \qquad T_{\mathrm{sw}} = 0.10~\text{s} + \kappa_g \Big(\frac{w}{10~\text{m/s}}\Big)^{3},$$
# with $\kappa = 0.30$ s for G1 and $0.05$ s for G2. The growth with the wind is an assumption of this example. No model contains this law: it represents an interface without a textbook formula.

# %%
import os, json, time, math, warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import lsq_linear
import matplotlib
if not os.environ.get("DISPLAY") and "ipykernel" not in __import__("sys").modules:
    matplotlib.use("Agg")                        # no window when run as a script on a machine without display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # fixed numbers of iterations are used on purpose


def nice_log(axis, subs=(1.0, 2.0, 5.0)):
    # log axis with plain labels (0.1, 0.2, 0.5, 1, 2, 5, ...) and no crowded minor labels
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


# "full": the 5 seeds of the article (about 12-18 min on a free Colab CPU). "quick": 1 seed (about 4 min).
MODE = "full"
MODE = os.environ.get("NB_MODE", MODE)           # the environment variable NB_MODE overrides the line above
SEEDS = [0] if MODE == "quick" else [0, 1, 2, 3, 4]
N_TRAIN = [20, 50, 150, 500]                     # numbers of measured snapshots
FIG_N = 150                                      # snapshots used in the bar panels and example curves of the figures

OUT = Path("results"); OUT.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d6", "lines.linewidth": 1.3, "savefig.bbox": "tight",
    "pdf.fonttype": 42})
COL_W = 3.5                                      # IEEE single-column width, inches
INK, MUTED, LIGHT = "#0b0b0b", "#6b6a66", "#a9a8a3"

print(f"MODE = {MODE}, seeds = {SEEDS}")
print(f"numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
      f"matplotlib {matplotlib.__version__}")

# %%
# Sketch of the scenario (Fig. 5 of the article)
def wavy(ax, p0, p1, amp=0.10, waves=7, **kw):
    # a wavy line from p0 to p1: the acoustic path
    t = np.linspace(0, 1, 200)
    x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
    nx, ny = -(p1[1] - p0[1]), p1[0] - p0[0]
    norm = np.hypot(nx, ny)
    off = amp * np.sin(2 * np.pi * waves * t)
    ax.plot(x + off * nx / norm, y + off * ny / norm, **kw)


def draw_buoy(ax, x, size, tilt_deg, label, note):
    hull = np.array([[-size, 0.0], [size, 0.0], [0.6 * size, -0.45 * size], [-0.6 * size, -0.45 * size]])
    mast = np.array([[0.0, 0.0], [0.0, 1.6 * size]])
    c, s = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
    rot = np.array([[c, -s], [s, c]])
    ax.add_patch(Polygon(hull @ rot.T + [x, 0.05], closed=True, color="#eda100", zorder=4))
    m = mast @ rot.T + [x, 0.05]
    ax.plot(m[:, 0], m[:, 1], color=INK, lw=1.0, zorder=4)
    ax.add_patch(Circle(m[1], 0.09, color=INK, zorder=5))
    ax.text(x, -0.55, label, ha="center", va="top", fontsize=7, weight="bold")
    ax.text(x, -0.95, note, ha="center", va="top", fontsize=6, color=MUTED)
    return m[1]


fig, ax = plt.subplots(figsize=(COL_W, 2.5))
ax.set_xlim(0, 10); ax.set_ylim(-3.3, 4.9); ax.axis("off")
ax.add_patch(Rectangle((0, -3.0), 10, 3.0, color="#e3eff8", lw=0, zorder=0))          # water
ax.add_patch(Rectangle((0, -3.3), 10, 0.3, color="#d8c9a8", lw=0, zorder=0))          # seabed
xs = np.linspace(0, 10, 300)
ax.plot(xs, 0.07 * np.sin(5 * xs), color="#2a78d6", lw=0.8, zorder=1)                 # sea surface
sensor = (1.3, -2.8)
ax.add_patch(Rectangle((sensor[0] - 0.25, sensor[1] - 0.12), 0.5, 0.3, color=INK, zorder=4))
ax.text(sensor[0] + 0.4, sensor[1] + 0.02, "seabed sensor", fontsize=6.5, va="center")
a1 = draw_buoy(ax, 4.0, 0.32, 14, "G1", "small, tilts")
a2 = draw_buoy(ax, 7.6, 0.45, 0, "G2", "stabilized antenna")
for b in (a1, a2):                                                                     # acoustic hops
    wavy(ax, (sensor[0], sensor[1] + 0.2), (b[0], -0.1), color="#2a78d6", lw=0.9)
ax.text(1.45, -1.3, "acoustic hop\n(sound)", fontsize=6, color="#2a78d6", ha="center", va="bottom")
sat = (6.2, 4.3)
ax.add_patch(Rectangle((sat[0] - 0.22, sat[1] - 0.16), 0.44, 0.32, color=INK, zorder=4))
ax.add_patch(Rectangle((sat[0] - 0.9, sat[1] - 0.07), 0.6, 0.14, color="#6b8fb3", zorder=4))
ax.add_patch(Rectangle((sat[0] + 0.3, sat[1] - 0.07), 0.6, 0.14, color="#6b8fb3", zorder=4))
ax.text(sat[0] + 1.05, sat[1], "LEO satellite\n(600 km)", fontsize=6.5, va="center")
for b in (a1, a2):                                                                     # satellite hops
    ax.annotate("", xy=(sat[0], sat[1] - 0.2), xytext=b,
                arrowprops=dict(arrowstyle="-|>", color="#eb6834", lw=0.9, shrinkA=2, shrinkB=2))
ax.text(4.25, 2.3, "satellite hop\n(radio)", fontsize=6, color="#eb6834", ha="center")
for yy in (2.9, 3.3):                                                                  # wind
    ax.annotate("", xy=(2.0, yy), xytext=(0.4, yy), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
ax.text(0.4, 3.6, "wind: wave noise + buoy tilt", fontsize=6, color=MUTED)
fig.savefig(FIG / "fig_example2_setup.pdf"); plt.show()

# %% [markdown]
# ## 2. Predictors
#
# All physical models start from the calibrated hop laws: each law has a few unknown numbers, fitted to the measurements by least squares within physical bounds. The *coupled* laws add the tilt loss of G1 in its small-tilt form, so that one wind measurement enters both hops. At prediction time, every model receives the same inputs: the gateway $g$, its distance $r$ to the sensor, the satellite elevation $\varepsilon$, and the wind speed $w$.
#
# | Model | Hop model | Junction at the buoy | Training data |
# |---|---|---|---|
# | MLP (data only) | none: $(g, r, \varepsilon, w)$ to end-to-end rate | none | end-to-end rates |
# | Hop MLPs + learned junction | one network per hop | learned $\varphi$ | per-hop SNRs and end-to-end rates |
# | Hop MLPs + TS + learned overhead | one network per hop | time sharing plus learned $T_{\mathrm{sw}}(g, w)$ | per-hop SNRs and end-to-end rates |
# | (Coupled) chain + min junction | calibrated laws | minimum | per-hop SNRs |
# | (Coupled) chain + time-sharing junction | calibrated laws | $R_aR_s/(R_a+R_s)$ | per-hop SNRs |
# | (Coupled) chain + learned junction | calibrated laws | learned $\varphi$ | per-hop SNRs and end-to-end rates |
# | Coupled chain + TS + constant, wind-linear, or wind-quadratic overhead | coupled laws | time sharing plus $T_{\mathrm{sw}} = t_0$, $t_0 + t_1 w$, or $t_0 + t_2 w^2$ per gateway | per-hop SNRs and end-to-end rates |
# | Coupled chain + TS + learned overhead | coupled laws | time sharing plus a network $T_{\mathrm{sw}}(g, w) \ge 0$ | per-hop SNRs and end-to-end rates |
#
# None of the overhead models has the cubic law that generates the data. The *nominal chain* (textbook laws, minimum junction, no data) is shown as a reference. The example is a composition of two segment models with a calibrated or learned part at the interface, and not a joint PINN that enforces the acoustic and electromagnetic equations together.

# %% [markdown]
# ## 3. Physics of the two hops
#
# **Acoustic hop (Section III-E).** The sensor lies at a depth of 100 m, and the buoy is at horizontal distance $r$, so the acoustic path is $l = \sqrt{r^2 + 0.1^2}$ km. The transmission loss follows Eq. (5) with the Thorp absorption at 12 kHz. The true ocean has $n_s = 1.7$, $\zeta = 1.25$, $\Delta_0 = 3$ dB, and the ripple of worked example 1, which no model contains. The noise is the usual ambient-noise model (turbulence, shipping, wind-driven surface noise, thermal noise) over a 4 kHz band, and the source level is 165 dB re 1 µPa at 1 m.
#
# **Satellite hop (Section III-A).** The satellite orbits at 600 km. The slant range follows Eq. (7) and the free-space loss Eq. (8) at 2 GHz. The link budget uses a 50 mW transmitter, a satellite $G/T$ of $-15$ dB/K, and a 10 kHz band:
# $$\mathrm{SNR}_s = P_t + G(\varepsilon) + G/T + 228.6 - 10\log_{10} B - \mathrm{FSPL}(d) - \frac{a}{\sin\varepsilon} - b - L_{\mathrm{tilt}}.$$
# G1 has a 3 dBi patch pointing up, whose gain falls as $\sin\varepsilon$, and G2 has a stabilized 8 dBi antenna. The excess loss has true values $a = 1$ dB and $b = 1.5$ dB. The tilt loss of G1 saturates in the true environment,
# $$L_{\mathrm{tilt}}^{\mathrm{true}} = 10\log_{10}\!\big(1 + k\, w^2 \cos^2\varepsilon\big), \qquad k = 0.0092,$$
# whereas the coupled chains use its small-tilt approximation $c\, w^2 \cos^2\varepsilon$, with $c$ fitted to the data. The coupled chains therefore carry a wrong form of the coupling law in a storm.
#
# **Rates.** Each hop rate is $B \log_2(1 + \mathrm{SNR}/\Gamma)$, with an SNR gap $\Gamma$ of 9 dB for the acoustic modem and 3 dB for the satellite modem.

# %%
# Geometry, wind and domains
DEPTH_KM = 0.1                                   # sensor depth
R_MIN, R_MAX, R_TRAIN = 0.3, 6.0, 3.0            # buoy distance (km): full domain, measured region r <= 3 km
E_MIN, E_MAX, E_TRAIN = 10.0, 90.0, 40.0         # satellite elevation (deg): full domain, measured region >= 40 deg
W_MIN, W_TRAIN, W_MAX = 2.0, 8.0, 14.0           # wind (m/s): measured 2-8 m/s, storm test 8-14 m/s

# Acoustic hop
F_KHZ, B_AC_HZ, SL_DB, GAP_AC_DB, SHIPPING = 12.0, 4000.0, 165.0, 9.0, 0.5
TRUE_AC = dict(n_s=1.7, zeta=1.25, delta0=3.0)   # true ocean (same as worked example 1)
NOM_AC = dict(n_s=1.5, zeta=1.0, delta0=0.0)     # textbook values
RIPPLE_DB, RIPPLE_PERIOD_KM, RIPPLE_DECAY_KM = 2.0, 1.5, 4.0

# Satellite hop
RE_M, H_SAT_M, C0 = 6371e3, 600e3, 299792458.0
FC_HZ, B_SAT_HZ, PT_DBW, GT_DBK, GAP_SAT_DB, BOLTZ_DB = 2e9, 10e3, -13.0, -15.0, 3.0, 228.6
G_ANT = {0: 3.0, 1: 8.0}                         # G1: 3 dBi patch (gain falls as sin(eps)); G2: stabilized 8 dBi
TRUE_SAT = dict(a=1.0, b=1.5)                    # true excess loss a/sin(eps) + b
TILT_K = 0.0092                                  # true tilt law: 10 log10(1 + k w^2 cos^2 eps), G1 only
TILT_C_SMALL = 10 * TILT_K / np.log(10)          # slope of its small-tilt approximation c w^2 cos^2 eps (about 0.04)
NOM_SAT = dict(a=0.0, b=0.0, tilt=0.0)           # textbook link budget

# Practical gateway: packet size and switching overhead T_sw = T0 + kappa_g (w / 10)^3
PACKET_KBIT = 2.048                              # 256-byte packets
T_SW0_S, KAPPA_S, W_REF = 0.10, {0: 0.30, 1: 0.05}, 10.0

NOISE_AC_DB, NOISE_SAT_DB = 1.5, 1.0             # measurement noise on each hop SNR (std, dB)
NOISE_E2E = 0.05                                 # independent error of the throughput measurement (5 %, log-normal)
MIN_RATE = 1e-9                                  # kbit/s, guards divisions by a dead hop


def thorp_db_per_km(f_khz):
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def ambient_noise_db(f_khz, wind, shipping=SHIPPING):
    # Power spectral density of ambient noise (dB re 1 uPa^2/Hz), f in kHz, wind in m/s:
    # turbulence + shipping + wind-driven surface noise + thermal noise
    lf = np.log10(f_khz)
    n_turb = 17 - 30 * lf
    n_ship = 40 + 20 * (shipping - 0.5) + 26 * lf - 60 * np.log10(f_khz + 0.03)
    n_wind = 50 + 7.5 * np.sqrt(wind) + 20 * lf - 40 * np.log10(f_khz + 0.4)
    n_th = -15 + 20 * lf
    return 10 * np.log10(10 ** (n_turb / 10) + 10 ** (n_ship / 10) + 10 ** (n_wind / 10) + 10 ** (n_th / 10))


def acoustic_path_km(r_km):
    return np.sqrt(np.asarray(r_km, dtype=float) ** 2 + DEPTH_KM ** 2)


def acoustic_tl_db(r_km, n_s, zeta, delta0):
    # Eq. (5) without the ripple, on the slant acoustic path
    l = acoustic_path_km(r_km)
    return 10 * n_s * np.log10(l / 1e-3) + zeta * l * thorp_db_per_km(F_KHZ) + delta0


def ripple_db(r_km):
    l = acoustic_path_km(r_km)
    return RIPPLE_DB * np.sin(2 * np.pi * l / RIPPLE_PERIOD_KM) * np.exp(-l / RIPPLE_DECAY_KM)


def noise_level_db(wind):
    return ambient_noise_db(F_KHZ, wind) + 10 * np.log10(B_AC_HZ)


def acoustic_snr_db(tl_db, wind):
    return SL_DB - tl_db - noise_level_db(wind)


def true_acoustic_snr_db(r_km, wind):
    return acoustic_snr_db(acoustic_tl_db(r_km, **TRUE_AC) + ripple_db(r_km), wind)


def slant_range_m(eps_deg):
    # Eq. (7) of the article
    e = np.radians(eps_deg)
    return np.sqrt(RE_M ** 2 * np.sin(e) ** 2 + H_SAT_M ** 2 + 2 * H_SAT_M * RE_M) - RE_M * np.sin(e)


def link_budget_db(eps_deg, g):
    # Known part of the satellite SNR: power, antenna gain, G/T, bandwidth and free-space loss (Eq. (8))
    e = np.radians(eps_deg)
    fspl = 20 * np.log10(4 * np.pi * slant_range_m(eps_deg) * FC_HZ / C0)
    gain = np.where(np.asarray(g) == 0, G_ANT[0] + 10 * np.log10(np.sin(e)), G_ANT[1])
    return PT_DBW + gain + GT_DBK + BOLTZ_DB - 10 * np.log10(B_SAT_HZ) - fspl


def tilt_regressor(eps_deg, g, wind):
    # w^2 cos^2(eps) for the small buoy G1, zero for the stabilized G2
    return np.where(np.asarray(g) == 0, np.asarray(wind, dtype=float) ** 2 * np.cos(np.radians(eps_deg)) ** 2, 0.0)


def satellite_snr_db(eps_deg, g, wind, a, b, tilt):
    # model form used by the calibrated chains: tilt loss = tilt * w^2 cos^2(eps) (small-tilt approximation)
    return link_budget_db(eps_deg, g) - a / np.sin(np.radians(eps_deg)) - b - tilt * tilt_regressor(eps_deg, g, wind)


def true_tilt_loss_db(eps_deg, g, wind):
    # true tilt loss of the environment: saturates for large tilts
    return 10 * np.log10(1 + TILT_K * tilt_regressor(eps_deg, g, wind))


def true_satellite_snr_db(eps_deg, g, wind):
    return (link_budget_db(eps_deg, g) - TRUE_SAT["a"] / np.sin(np.radians(eps_deg)) - TRUE_SAT["b"]
            - true_tilt_loss_db(eps_deg, g, wind))


def rate_kbps(snr_db, bandwidth_hz, gap_db):
    return bandwidth_hz * np.log2(1 + 10 ** ((np.asarray(snr_db) - gap_db) / 10)) / 1e3


def acoustic_rate(snr_db):
    return rate_kbps(snr_db, B_AC_HZ, GAP_AC_DB)


def satellite_rate(snr_db):
    return rate_kbps(snr_db, B_SAT_HZ, GAP_SAT_DB)


# gateway behaviors: every junction takes the two hop rates, the gateway and the wind
def join_full_duplex(r_a, r_s, g=None, wind=None):
    return np.minimum(r_a, r_s)


def join_half_duplex(r_a, r_s, g=None, wind=None):
    # best time split between receiving and transmitting: tau R_a = (1 - tau) R_s
    return r_a * r_s / np.maximum(r_a + r_s, MIN_RATE)


def join_with_overhead(r_a, r_s, overhead_s):
    # time sharing plus a per-packet overhead (s): R = L / (L/R_a + L/R_s + T)
    t = (PACKET_KBIT / np.maximum(r_a, MIN_RATE) + PACKET_KBIT / np.maximum(r_s, MIN_RATE)
         + np.maximum(overhead_s, 0.0))
    return PACKET_KBIT / t


def true_switch_overhead_s(g, wind):
    kappa = np.where(np.asarray(g) == 0, KAPPA_S[0], KAPPA_S[1])
    return T_SW0_S + kappa * (np.asarray(wind, dtype=float) / W_REF) ** 3


def join_practical(r_a, r_s, g, wind):
    return join_with_overhead(r_a, r_s, true_switch_overhead_s(g, wind))


SCENARIOS = {"practical": join_practical, "half-duplex": join_half_duplex, "full-duplex": join_full_duplex}
SCN_TITLE = {"practical": "practical half-duplex gateway", "half-duplex": "ideal half-duplex gateway",
             "full-duplex": "full-duplex gateway"}


def true_rates(g, r, eps, wind):
    # noise-free per-hop rates of the true environment
    return (acoustic_rate(true_acoustic_snr_db(r, wind)),
            satellite_rate(true_satellite_snr_db(eps, g, wind)))


def true_e2e(g, r, eps, wind, scenario):
    return SCENARIOS[scenario](*true_rates(g, r, eps, wind), g, wind)


print(f"Thorp absorption at 12 kHz: {thorp_db_per_km(12.0):.2f} dB/km")
print(f"Slant range to a 600 km LEO: {slant_range_m(90) / 1e3:.0f} km at 90 deg, {slant_range_m(10) / 1e3:.0f} km at 10 deg")
for ww in (5.0, 8.0, 11.0, 14.0):
    print(f"Tilt loss of G1 at 40 deg, wind {ww:4.1f} m/s: true {float(true_tilt_loss_db(40, 0, ww)):4.2f} dB, "
          f"small-tilt approximation {float(TILT_C_SMALL * tilt_regressor(40, 0, ww)):4.2f} dB")
for ww in (2.0, 5.0, 8.0, 12.0, 14.0):
    print(f"Switching overhead at {ww:4.1f} m/s: G1 {float(true_switch_overhead_s(0, ww)):.3f} s, "
          f"G2 {float(true_switch_overhead_s(1, ww)):.3f} s")

# %% [markdown]
# ## 4. Measurements and test sets
#
# A snapshot is one gateway (G1 or G2, at random), one buoy distance, one elevation, and one wind speed. For each training snapshot, the buoy logs the SNR of each hop (error of 1.5 dB acoustic, 1 dB satellite) and the end-to-end throughput, with its own error of 5%.
#
# - The **prediction error** is the RMSE of the end-to-end rate against the noise-free truth, in kbit/s.
# - In a **gateway choice**, each model picks the buoy with the higher predicted rate. The **throughput loss** is the share of the best possible throughput that its choices lose.
# - Two **single-segment rules** are given for comparison: *nearest buoy* (acoustic hop only) and *always G2* (satellite hop only).
#
# The three gateway behaviors use the same snapshots and the same SNR errors, so the calibrated hop laws are identical in the three scenarios.

# %%
def sample_box(n, rng, r, e, w):
    # n snapshots: gateway, distance (km), elevation (deg), wind (m/s), uniform in the given ranges
    g = rng.integers(0, 2, n)
    return g, rng.uniform(r[0], r[1], n), rng.uniform(e[0], e[1], n), rng.uniform(w[0], w[1], n)


def sample_snapshots(n, rng, w_hi=None):
    # training snapshots in the measured conditions
    return sample_box(n, rng, (R_MIN, R_TRAIN), (E_TRAIN, E_MAX), (W_MIN, W_TRAIN if w_hi is None else w_hi))


def measure(g, r, eps, wind, rng, scenario):
    # per-hop SNRs with measurement errors; end-to-end throughput delivered with these SNRs, counted with its own error
    snr_a = true_acoustic_snr_db(r, wind) + rng.normal(0, NOISE_AC_DB, len(r))
    snr_s = true_satellite_snr_db(eps, g, wind) + rng.normal(0, NOISE_SAT_DB, len(r))
    y = SCENARIOS[scenario](acoustic_rate(snr_a), satellite_rate(snr_s), g, wind)
    return snr_a, snr_s, y * np.exp(rng.normal(0, NOISE_E2E, len(r)))


# prediction tests
TEST = sample_box(8000, np.random.default_rng(123), (R_MIN, R_MAX), (E_MIN, E_MAX), (W_MIN, W_TRAIN))
TEST_IN = (TEST[1] <= R_TRAIN) & (TEST[2] >= E_TRAIN)
STORM = sample_box(4000, np.random.default_rng(789), (R_MIN, R_TRAIN), (E_TRAIN, E_MAX), (W_TRAIN, W_MAX))
BOTH = sample_box(4000, np.random.default_rng(790), (R_TRAIN, R_MAX), (E_MIN, E_TRAIN), (W_TRAIN, W_MAX))
PRED_SETS = {"main": TEST, "storm": STORM, "both": BOTH}


# gateway-choice tests: G1 at distance r1, G2 at distance r2, same elevation and wind
def make_selection_set(seed, r, e, w, n=4000):
    rng = np.random.default_rng(seed)
    return dict(r1=rng.uniform(r[0], r[1], n), r2=rng.uniform(r[0], r[1], n),
                eps=rng.uniform(e[0], e[1], n), wind=rng.uniform(w[0], w[1], n))


SEL_SETS = {"main": make_selection_set(456, (R_MIN, R_MAX), (E_MIN, E_MAX), (W_MIN, W_TRAIN)),
            "storm": make_selection_set(457, (R_MIN, R_TRAIN), (E_TRAIN, E_MAX), (W_TRAIN, W_MAX)),
            "both": make_selection_set(458, (R_TRAIN, R_MAX), (E_MIN, E_TRAIN), (W_TRAIN, W_MAX))}
SEL_IN = ((SEL_SETS["main"]["r1"] <= R_TRAIN) & (SEL_SETS["main"]["r2"] <= R_TRAIN)
          & (SEL_SETS["main"]["eps"] >= E_TRAIN))
TESTS = {"geometry": "new geometry", "storm": "storm", "both": "both shifts"}      # names used in tables and figures

# noise-free truths, computed once per scenario
TRUTH = {scn: {k: true_e2e(*v, scn) for k, v in PRED_SETS.items()} for scn in SCENARIOS}
SEL_TRUE = {}
for scn in SCENARIOS:
    SEL_TRUE[scn] = {}
    for k, s in SEL_SETS.items():
        n = len(s["r1"])
        SEL_TRUE[scn][k] = (true_e2e(np.zeros(n, dtype=int), s["r1"], s["eps"], s["wind"], scn),
                            true_e2e(np.ones(n, dtype=int), s["r2"], s["eps"], s["wind"], scn))

print(f"Prediction tests: {TEST_IN.sum()} inside, {(~TEST_IN).sum()} new geometry, {len(STORM[0])} storm, "
      f"{len(BOTH[0])} both shifts")

# share of snapshots in which the acoustic hop is the slower one (noise-free truth)
BOTTLENECK = {}
for label, snaps, mask in (("inside", TEST, TEST_IN), ("new geometry", TEST, ~TEST_IN),
                           ("storm", STORM, None), ("both shifts", BOTH, None)):
    ra_, rs_ = true_rates(*snaps)
    mask = np.ones(len(ra_), dtype=bool) if mask is None else mask
    for gw, name in ((0, "G1"), (1, "G2")):
        m = mask & (snaps[0] == gw)
        BOTTLENECK[f"{name} {label}"] = float(np.mean(ra_[m] < rs_[m]))
print("Share of snapshots limited by the acoustic hop:", {k: round(v, 2) for k, v in BOTTLENECK.items()})


def choice_scores(choice, scn, setname, mask=None):
    # throughput loss (%) and share of wrong choices (%)
    r1, r2 = SEL_TRUE[scn][setname]
    mask = np.ones(len(r1), dtype=bool) if mask is None else mask
    best, got = np.maximum(r1, r2), np.where(choice == 0, r1, r2)
    return (float(100 * (1 - got[mask].sum() / best[mask].sum())),
            float(100 * np.mean(got[mask] < best[mask] - 1e-9)))


def all_choice_scores(choices, scn):
    out = {}
    out["loss_inside"], out["wrong_inside"] = choice_scores(choices["main"], scn, "main", SEL_IN)
    out["loss_geometry"], out["wrong_geometry"] = choice_scores(choices["main"], scn, "main", ~SEL_IN)
    out["loss_storm"], out["wrong_storm"] = choice_scores(choices["storm"], scn, "storm")
    out["loss_both"], out["wrong_both"] = choice_scores(choices["both"], scn, "both")
    return out


RULE_FUNCS = {"Nearest buoy": lambda s: (s["r2"] < s["r1"]).astype(int),
              "Always G2": lambda s: np.ones(len(s["r1"]), dtype=int)}
RULES = {scn: {name: all_choice_scores({k: f(s) for k, s in SEL_SETS.items()}, scn)
               for name, f in RULE_FUNCS.items()} for scn in SCENARIOS}
for scn in SCENARIOS:
    print(f"[{scn}] throughput loss of the single-segment rules (%): " + "; ".join(
        f"{name}: geometry {v['loss_geometry']:.1f}, storm {v['loss_storm']:.1f}, both {v['loss_both']:.1f}"
        for name, v in RULES[scn].items()))

# %% [markdown]
# ## 5. Models
#
# **Networks.** Every network is a multilayer perceptron with two hidden layers and $\tanh$ activations, trained with full-batch Adam (learning rate $2\cdot10^{-3}$, 3000 iterations, weight decay $10^{-4}$). Inputs are scaled to $[0, 1]$ with fixed physical bounds.
#
# **Calibrated hop laws, within physical bounds.** Once the known noise is removed, the acoustic law is linear in $(n_s, \zeta, \Delta_0)$, and the excess satellite loss is linear in $(a, b)$, plus $c$ for the coupled law of G1. These numbers are fitted by bounded least squares.
#
# | Parameter | Meaning | Bound | Reason |
# |---|---|---|---|
# | $n_s$ | spreading factor | $1 \le n_s \le 2$ | between cylindrical and spherical spreading |
# | $\zeta$ | scale of the Thorp absorption | $\zeta \ge 0$ | water absorbs energy |
# | $\Delta_0$ | fixed offset in dB | free | unknown source and receiver details |
# | $a, b$ | excess losses of the radio path | $\ge 0$ | an extra loss cannot add power |
# | $c$ | tilt coefficient of G1 | $\ge 0$ | stronger wind cannot improve the pointing |
# | $t_0, t_1, t_2$ | switching-delay coefficients | $\ge 0$ | a delay is positive and does not decrease with the wind |
#
# The bounds use only textbook physics, never the true values of the simulator. The unbounded twin of every closed-form model (plain least squares) is also run and audited.
#
# **Learned junction $\varphi$.** The end-to-end rate is $\hat R = \min(\hat R_a, \hat R_s)\,\varphi(\rho)$ with $\rho = \min(\hat R_a, \hat R_s)/\max(\hat R_a, \hat R_s)$, and a small network learns $\varphi \in [0, 1]$.
#
# **Overhead models.** For each training snapshot, the delay implied by the measurements is $\hat T_i = L/y_i - L/\hat R_{a,i} - L/\hat R_{s,i}$. Four forms are fitted to these values from $(g, w)$: a constant, a straight line, a parabola (per gateway, bounded least squares), and a network with output $\ge 0$.

# %%
LR, ITERS, Y_SCALE, SNR_SCALE, T_SCALE = 2e-3, 3000, 20.0, 20.0, 0.5    # Adam settings and output scales
BOUNDS_AC = (np.array([1.0, 0.0, -np.inf]), np.array([2.0, np.inf, np.inf]))   # (n_s, zeta, delta0)


def make_mlp(width, n, seed):
    # Full-batch Adam for a fixed number of iterations; random_state fixes the initial weights.
    return MLPRegressor(hidden_layer_sizes=(width, width), activation="tanh", solver="adam",
                        learning_rate_init=LR, max_iter=ITERS, batch_size=int(n), tol=0.0,
                        n_iter_no_change=10 ** 9, alpha=1e-4, random_state=seed)


def scale(x, lo, hi):
    return (np.asarray(x, dtype=float) - lo) / (hi - lo)


def fit_ls(A, y, lower=None, upper=None):
    # Least squares. Without bounds: plain least squares. With bounds: bounded least squares (BVLS),
    # on scaled columns so that the solver is well conditioned; the coefficients are scaled back.
    if lower is None:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    p = A.shape[1]
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (p,))
    upper = np.full(p, np.inf) if upper is None else np.broadcast_to(np.asarray(upper, dtype=float), (p,))
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    res = lsq_linear(A / norms, y, bounds=(lower * norms, upper * norms), method="bvls", tol=1e-12, max_iter=2000)
    return np.clip(res.x / norms, lower, upper)


# hop models: return (R_a, R_s) in kbit/s
class CalibratedHops:
    # Per-segment physical laws fitted by least squares, within physical bounds (bounded=True) or not.
    def __init__(self, data=None, coupled=False, fit=True, bounded=True):
        self.coupled, self.bounded = coupled, bounded
        if not fit:                                     # nominal (textbook) laws
            self.ac = np.array([NOM_AC["n_s"], NOM_AC["zeta"], NOM_AC["delta0"]])
            self.sat = {0: np.array([NOM_SAT["a"], NOM_SAT["b"], 0.0]), 1: np.array([NOM_SAT["a"], NOM_SAT["b"], 0.0])}
            return
        g, r, eps, wind, snr_a, snr_s = (data[k] for k in ("g", "r", "eps", "wind", "snr_a", "snr_s"))
        # Acoustic hop: TL = SL - noise(wind) - SNR is linear in (n_s, zeta, delta0)
        l = acoustic_path_km(r)
        A = np.stack([10 * np.log10(l / 1e-3), l * thorp_db_per_km(F_KHZ), np.ones_like(l)], axis=1)
        target = SL_DB - noise_level_db(wind) - snr_a
        if len(r) >= 4:
            self.ac = fit_ls(A, target, *(BOUNDS_AC if bounded else (None, None)))
        else:                                           # too few snapshots: textbook slopes, fitted offset
            slopes = np.array([NOM_AC["n_s"], NOM_AC["zeta"]])
            self.ac = np.array([*slopes, float(np.mean(target - A[:, :2] @ slopes))])
        # Satellite hop of each gateway: excess loss = budget - SNR, linear in (a, b) [and c for the coupled G1]
        self.sat = {}
        for k in (0, 1):
            m = g == k
            excess = link_budget_db(eps[m], k) - snr_s[m]
            cols = [1 / np.sin(np.radians(eps[m])), np.ones(m.sum())]
            if coupled and k == 0:
                cols.append(tilt_regressor(eps[m], k, wind[m]))
            if m.sum() >= len(cols) + 1:
                coef = fit_ls(np.stack(cols, axis=1), excess, 0.0 if bounded else None)
            else:                                       # too few snapshots of this gateway: constant loss only
                coef = np.array([0.0, float(excess.mean()) if m.sum() else 0.0])
                coef = np.maximum(coef, 0.0) if bounded else coef
            self.sat[k] = np.concatenate([coef, np.zeros(3 - len(coef))])     # (a, b, c); c = 0 if not fitted

    def params(self):
        return dict(n_s=float(self.ac[0]), zeta=float(self.ac[1]), delta0=float(self.ac[2]),
                    a_G1=float(self.sat[0][0]), b_G1=float(self.sat[0][1]), c_G1=float(self.sat[0][2]),
                    a_G2=float(self.sat[1][0]), b_G2=float(self.sat[1][1]))

    def outside_physics(self, tol=1e-9):
        # parameters outside their physical range (used to audit the unbounded fits)
        p = self.params()
        bad = [] if 1.0 - tol <= p["n_s"] <= 2.0 + tol else ["n_s"]
        bad += [k for k in ("zeta", "a_G1", "b_G1", "c_G1", "a_G2", "b_G2") if p[k] < -tol]
        return bad

    def at_bound(self, tol=1e-6):
        # parameters sitting on a bound (for the bounded fits; a fitted c of the uncoupled law is always 0, skipped)
        p = self.params()
        hit = [k for k, lim in (("n_s", 1.0), ("n_s", 2.0)) if abs(p[k] - lim) < tol]
        keys = ["zeta", "a_G1", "b_G1", "a_G2", "b_G2"] + (["c_G1"] if self.coupled else [])
        return hit + [k for k in keys if abs(p[k]) < tol]

    def snrs(self, g, r, eps, wind):
        g = np.asarray(g)
        snr_a = acoustic_snr_db(acoustic_tl_db(r, *self.ac), wind)
        coef = np.where(g[:, None] == 0, self.sat[0][None, :], self.sat[1][None, :])
        snr_s = (link_budget_db(eps, g) - coef[:, 0] / np.sin(np.radians(eps)) - coef[:, 1]
                 - coef[:, 2] * tilt_regressor(eps, g, wind))
        return snr_a, snr_s

    def hop_rates(self, g, r, eps, wind):
        snr_a, snr_s = self.snrs(g, r, eps, wind)
        return acoustic_rate(snr_a), satellite_rate(snr_s)


class HopMLPs:
    # Data only, with the same per-hop measurements as the calibrated laws.
    def __init__(self, data, seed):
        n = len(data["r"])
        self.ac = make_mlp(64, n, seed).fit(self._xa(data["r"], data["wind"]), data["snr_a"] / SNR_SCALE)
        self.sa = make_mlp(64, n, seed).fit(self._xs(data["g"], data["eps"], data["wind"]), data["snr_s"] / SNR_SCALE)

    @staticmethod
    def _xa(r, wind):
        return np.stack([scale(r, R_MIN, R_MAX), scale(wind, W_MIN, W_MAX)], axis=1)

    @staticmethod
    def _xs(g, eps, wind):
        return np.stack([np.asarray(g, dtype=float), scale(eps, E_MIN, E_MAX), scale(wind, W_MIN, W_MAX)], axis=1)

    def hop_rates(self, g, r, eps, wind):
        return (acoustic_rate(self.ac.predict(self._xa(r, wind)) * SNR_SCALE),
                satellite_rate(self.sa.predict(self._xs(g, eps, wind)) * SNR_SCALE))


# junctions
class LearnedJunction:
    # phi(rho) learned from the end-to-end rates, bounded to [0, 1]
    MIN_FIT_RATE = 0.05                              # kbit/s: snapshots with a dead hop carry no information on phi

    def __init__(self, hops, data, seed):
        r_a, r_s = hops.hop_rates(data["g"], data["r"], data["eps"], data["wind"])
        lo, rho = self.lo_rho(r_a, r_s)
        ok = lo > self.MIN_FIT_RATE
        self.m = make_mlp(16, ok.sum(), seed).fit(rho[ok, None], data["y"][ok] / lo[ok]) if ok.sum() >= 3 else None

    @staticmethod
    def lo_rho(r_a, r_s):
        lo, hi = np.minimum(r_a, r_s), np.maximum(r_a, r_s)
        return lo, lo / np.maximum(hi, MIN_RATE)

    def phi(self, rho):
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        if self.m is None:
            return np.ones_like(rho)
        return np.clip(self.m.predict(rho[:, None]), 0.0, 1.0)

    def __call__(self, r_a, r_s, g=None, wind=None):
        lo, rho = self.lo_rho(r_a, r_s)
        return lo * self.phi(rho)


class OverheadJunction:
    # Time sharing plus a per-packet delay T(g, w): "const", "linear" (t0 + t1 w), "quad" (t0 + t2 w^2) or "net"
    MIN_FIT_RATE = 0.5                               # kbit/s

    def __init__(self, hops, data, form, seed=0, bounded=True):
        self.form = form
        g, wind, y = data["g"], data["wind"], data["y"]
        r_a, r_s = hops.hop_rates(g, data["r"], data["eps"], wind)
        ok = (y > self.MIN_FIT_RATE) & (r_a > self.MIN_FIT_RATE) & (r_s > self.MIN_FIT_RATE)
        t = (PACKET_KBIT / np.maximum(y, MIN_RATE) - PACKET_KBIT / np.maximum(r_a, MIN_RATE)
             - PACKET_KBIT / np.maximum(r_s, MIN_RATE))
        g_ok, w_ok, t_ok = g[ok], wind[ok], t[ok]
        self.coef, self.m = {}, None
        if form in ("const", "linear", "quad"):
            for k in (0, 1):
                m = g_ok == k
                if form in ("linear", "quad") and m.sum() >= 3:
                    X = np.stack([np.ones(m.sum()), self._basis(w_ok[m])], axis=1)
                    self.coef[k] = fit_ls(X, t_ok[m], 0.0 if bounded else None)
                else:
                    v = float(t_ok[m].mean()) if m.sum() else 0.0
                    self.coef[k] = np.array([max(v, 0.0) if bounded else v, 0.0])
        elif ok.sum() >= 3:
            self.m = make_mlp(16, ok.sum(), seed).fit(self._x(g_ok, w_ok), t_ok / T_SCALE)

    @staticmethod
    def _x(g, wind):
        return np.stack([np.asarray(g, dtype=float), scale(wind, W_MIN, W_MAX)], axis=1)

    def _basis(self, wind):
        return wind ** 2 if self.form == "quad" else wind

    def overhead(self, g, wind):
        g = np.atleast_1d(np.asarray(g)); wind = np.atleast_1d(np.asarray(wind, dtype=float))
        if self.form == "net":
            t = np.zeros_like(wind) if self.m is None else self.m.predict(self._x(g, wind)) * T_SCALE
        else:
            c = np.where(g[:, None] == 0, self.coef[0][None, :], self.coef[1][None, :])
            t = c[:, 0] + c[:, 1] * self._basis(wind)
        return np.maximum(t, 0.0)

    def __call__(self, r_a, r_s, g, wind):
        return join_with_overhead(r_a, r_s, self.overhead(g, wind))


class Composite:
    # hop model + junction = end-to-end predictor
    def __init__(self, hops, junction):
        self.hops, self.junction = hops, junction

    def predict(self, g, r, eps, wind):
        return np.maximum(self.junction(*self.hops.hop_rates(g, r, eps, wind), g, wind), 0.0)


class DataMLP:
    # Data only: (gateway, distance, elevation, wind) -> end-to-end rate
    def __init__(self, data, seed):
        self.m = make_mlp(64, len(data["y"]), seed).fit(self._x(data["g"], data["r"], data["eps"], data["wind"]),
                                                        data["y"] / Y_SCALE)

    @staticmethod
    def _x(g, r, eps, wind):
        return np.stack([np.asarray(g, dtype=float), scale(r, R_MIN, R_MAX), scale(eps, E_MIN, E_MAX),
                         scale(wind, W_MIN, W_MAX)], axis=1)

    def predict(self, g, r, eps, wind):
        return np.maximum(self.m.predict(self._x(g, r, eps, wind)) * Y_SCALE, 0.0)


KINDS = ["MLP", "HopMLP+J", "HopMLP+Tnet", "Chain+min", "Chain+TS", "Chain+J",
         "Coupled+min", "Coupled+TS", "Coupled+J", "Coupled+T0", "Coupled+Tw", "Coupled+Tq", "Coupled+Tnet"]
CLOSED_FORM = ["Chain+min", "Chain+TS", "Coupled+min", "Coupled+TS", "Coupled+T0", "Coupled+Tw", "Coupled+Tq"]
LABELS = {"MLP": "MLP (data only)",
          "HopMLP+J": "Hop MLPs + learned junction",
          "HopMLP+Tnet": "Hop MLPs + TS + learned overhead",
          "Chain+min": "Chain + min junction",
          "Chain+TS": "Chain + time-sharing junction",
          "Chain+J": "Chain + learned junction",
          "Coupled+min": "Coupled chain + min junction",
          "Coupled+TS": "Coupled chain + time-sharing junction",
          "Coupled+J": "Coupled chain + learned junction",
          "Coupled+T0": "Coupled chain + TS + constant overhead",
          "Coupled+Tw": "Coupled chain + TS + wind-linear overhead",
          "Coupled+Tq": "Coupled chain + TS + wind-quadratic overhead",
          "Coupled+Tnet": "Coupled chain + TS + learned overhead",
          "Nominal": "Nominal chain"}


def closed_form_models(cal, cpl, data, bounded):
    # the models that need no network: calibrated laws with an analytical junction or a fitted delay
    return {"Chain+min": Composite(cal, join_full_duplex),
            "Chain+TS": Composite(cal, join_half_duplex),
            "Coupled+min": Composite(cpl, join_full_duplex),
            "Coupled+TS": Composite(cpl, join_half_duplex),
            "Coupled+T0": Composite(cpl, OverheadJunction(cpl, data, "const", bounded=bounded)),
            "Coupled+Tw": Composite(cpl, OverheadJunction(cpl, data, "linear", bounded=bounded)),
            "Coupled+Tq": Composite(cpl, OverheadJunction(cpl, data, "quad", bounded=bounded))}


def build_models(data, seed):
    cal, cpl = CalibratedHops(data), CalibratedHops(data, coupled=True)                  # bounded (main)
    hop_mlp = HopMLPs(data, seed)
    models = {"MLP": DataMLP(data, seed),
              "HopMLP+J": Composite(hop_mlp, LearnedJunction(hop_mlp, data, seed)),
              "HopMLP+Tnet": Composite(hop_mlp, OverheadJunction(hop_mlp, data, "net", seed)),
              "Chain+J": Composite(cal, LearnedJunction(cal, data, seed)),
              "Coupled+J": Composite(cpl, LearnedJunction(cpl, data, seed)),
              "Coupled+Tnet": Composite(cpl, OverheadJunction(cpl, data, "net", seed)),
              **closed_form_models(cal, cpl, data, bounded=True)}
    models = {k: models[k] for k in KINDS}
    cal_u, cpl_u = CalibratedHops(data, bounded=False), CalibratedHops(data, coupled=True, bounded=False)
    unbounded = closed_form_models(cal_u, cpl_u, data, bounded=False)                   # audit twin
    return models, unbounded, dict(chain=cal, coupled=cpl, chain_unbounded=cal_u, coupled_unbounded=cpl_u)


NOMINAL = Composite(CalibratedHops(fit=False), join_full_duplex)


def rmse(pred, truth, mask=None):
    mask = np.ones(len(truth), dtype=bool) if mask is None else mask
    return float(np.sqrt(np.mean((pred[mask] - truth[mask]) ** 2)))


def choose(model, sel):
    # 0 = G1, 1 = G2: the gateway with the higher predicted end-to-end rate
    n = len(sel["r1"])
    p1 = model.predict(np.zeros(n, dtype=int), sel["r1"], sel["eps"], sel["wind"])
    p2 = model.predict(np.ones(n, dtype=int), sel["r2"], sel["eps"], sel["wind"])
    return (p2 > p1).astype(int)


def evaluate(model, scn):
    pred = {k: model.predict(*v) for k, v in PRED_SETS.items()}
    for p in pred.values():
        assert np.all(np.isfinite(p)) and np.all(p >= 0), "a prediction is not a finite, non-negative rate"
    t = TRUTH[scn]
    out = dict(rmse_inside=rmse(pred["main"], t["main"], TEST_IN), rmse_geometry=rmse(pred["main"], t["main"], ~TEST_IN),
               rmse_storm=rmse(pred["storm"], t["storm"]), rmse_both=rmse(pred["both"], t["both"]))
    out.update(all_choice_scores({k: choose(model, s) for k, s in SEL_SETS.items()}, scn))
    return out


NOMINAL_SCORES = {scn: evaluate(NOMINAL, scn) for scn in SCENARIOS}
for scn, v in NOMINAL_SCORES.items():
    print(f"[{scn}] nominal chain RMSE (kbit/s): inside {v['rmse_inside']:.2f}, new geometry {v['rmse_geometry']:.2f}, "
          f"storm {v['rmse_storm']:.2f}, both shifts {v['rmse_both']:.2f}")

# %% [markdown]
# ## 6. Check of the pipeline
#
# The code is first checked on noise-free data over all conditions, without the ripple.
#
# 1. Two hops of 10 kbit/s give 10 (full duplex), 5 (time sharing), and $10/3$ kbit/s with a delay of $L/10$ s.
# 2. The calibration, bounded and unbounded, recovers $n_s = 1.7$, $\zeta = 1.25$, $\Delta_0 = 3$ dB, $a = 1$ dB, $b = 1.5$ dB.
# 3. The coupled calibration recovers $c = 0.04$ from data generated with its own small-tilt form.
# 4. Given the exact hop rates of an ideal half-duplex gateway, the learned junction recovers $\varphi(\rho) = 1/(1+\rho)$.
# 5. Given the exact hop rates of the practical gateway, the learned overhead recovers $T_{\mathrm{sw}}(g, w)$ over the winds it has seen.

# %%
ten = np.array([10.0])
assert np.isclose(join_full_duplex(ten, ten)[0], 10.0) and np.isclose(join_half_duplex(ten, ten)[0], 5.0)
assert np.isclose(join_with_overhead(ten, ten, np.array([PACKET_KBIT / 10]))[0], 10 / 3)

_rng = np.random.default_rng(0)
_g, _r, _e, _w = sample_box(400, _rng, (R_MIN, R_MAX), (E_MIN, E_MAX), (W_MIN, W_MAX))
_sa = acoustic_snr_db(acoustic_tl_db(_r, **TRUE_AC), _w)                                    # no ripple, no noise
_ss_notilt = satellite_snr_db(_e, _g, _w, a=TRUE_SAT["a"], b=TRUE_SAT["b"], tilt=0.0)
_ss_tilt = satellite_snr_db(_e, _g, _w, a=TRUE_SAT["a"], b=TRUE_SAT["b"], tilt=0.04)         # model's own form
_base = dict(g=_g, r=_r, eps=_e, wind=_w, snr_a=_sa)
SANITY = {}
for bounded in (True, False):
    _cal = CalibratedHops(dict(_base, snr_s=_ss_notilt), bounded=bounded)
    _cpl = CalibratedHops(dict(_base, snr_s=_ss_tilt), coupled=True, bounded=bounded)
    p = _cal.params()
    assert np.allclose([p["n_s"], p["zeta"], p["delta0"], p["a_G1"], p["b_G1"], p["a_G2"], p["b_G2"]],
                       [1.7, 1.25, 3.0, 1.0, 1.5, 1.0, 1.5], atol=1e-6), p
    assert np.isclose(_cpl.params()["c_G1"], 0.04, atol=1e-6)
    SANITY[f"calibration_{'bounded' if bounded else 'unbounded'}"] = {k: round(v, 4) for k, v in p.items()}
    SANITY[f"c_G1_{'bounded' if bounded else 'unbounded'}"] = round(_cpl.params()["c_G1"], 4)
print("Calibration on clean data, bounded and unbounded: exact (n_s 1.7, zeta 1.25, delta0 3, a 1, b 1.5, c 0.04)")

_y = join_half_duplex(*_cal.hop_rates(_g, _r, _e, _w))
_junc = LearnedJunction(_cal, dict(_base, snr_s=_ss_notilt, y=_y), seed=0)
_rho = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
SANITY["phi_learned"] = [round(float(v), 3) for v in _junc.phi(_rho)]
SANITY["phi_true"] = [round(float(v), 3) for v in 1 / (1 + _rho)]
err = float(np.max(np.abs(np.array(SANITY["phi_learned"]) - np.array(SANITY["phi_true"]))))
print(f"Junction phi(rho) at rho = {_rho.tolist()}: learned {SANITY['phi_learned']}, true {SANITY['phi_true']}"
      f"  -> {'OK' if err < 0.03 else 'CHECK'} (max error {err:.3f})")

_cpl = CalibratedHops(dict(_base, snr_s=_ss_tilt), coupled=True)
_yp = join_practical(*_cpl.hop_rates(_g, _r, _e, _w), _g, _w)
_ovh = OverheadJunction(_cpl, dict(_base, snr_s=_ss_tilt, y=_yp), "net", seed=0)
_wq = np.array([3.0, 6.0, 9.0, 12.0])
for k, name in ((0, "G1"), (1, "G2")):
    SANITY[f"overhead_learned_{name}"] = [round(float(v), 3) for v in _ovh.overhead(np.full(4, k), _wq)]
    SANITY[f"overhead_true_{name}"] = [round(float(v), 3) for v in true_switch_overhead_s(np.full(4, k), _wq)]
    err = float(np.max(np.abs(np.array(SANITY[f"overhead_learned_{name}"]) - np.array(SANITY[f"overhead_true_{name}"]))))
    print(f"Delay of {name} at wind {_wq.tolist()} m/s: learned {SANITY[f'overhead_learned_{name}']}, "
          f"true {SANITY[f'overhead_true_{name}']} s  -> {'OK' if err < 0.03 else 'CHECK'} (max error {err:.3f} s)")
print("Pipeline checks done.")

# %% [markdown]
# ## 7. Experiment
#
# For each gateway behavior, number of snapshots $N$, and seed, all predictors use the same snapshots and are tested on the same points: 13 predictors with bounded calibration, plus the unbounded twins of the 7 closed-form ones. In `full` mode this is $3 \times 4 \times 5 = 60$ training sets.

# %%
OVH_WINDS = [3.0, 6.0, 9.0, 12.0]                  # winds at which the fitted delays are recorded
t0 = time.time()
RUNS, PARAMS, KEEP = [], [], {}
for scn in SCENARIOS:
    for n in N_TRAIN:
        for s in SEEDS:
            rng = np.random.default_rng(s)
            g, r, eps, wind = sample_snapshots(n, rng)
            snr_a, snr_s, y = measure(g, r, eps, wind, rng, scn)
            data = dict(g=g, r=r, eps=eps, wind=wind, snr_a=snr_a, snr_s=snr_s, y=y)
            models, unbounded, hops = build_models(data, s)
            for arm, group in (("bounded", models), ("unbounded", unbounded)):
                for kind, m in group.items():
                    run = dict(scenario=scn, arm=arm, model=kind, n=n, seed=s, **evaluate(m, scn))
                    j = getattr(m, "junction", None)
                    if isinstance(j, LearnedJunction):
                        run["phi"] = [float(v) for v in j.phi([0.25, 0.5, 0.75, 1.0])]
                    if isinstance(j, OverheadJunction):
                        run["overhead_G1"] = [float(v) for v in j.overhead(np.zeros(4, dtype=int), OVH_WINDS)]
                        run["overhead_G2"] = [float(v) for v in j.overhead(np.ones(4, dtype=int), OVH_WINDS)]
                        if j.coef:
                            run["overhead_coef"] = {str(k): [float(c) for c in v] for k, v in j.coef.items()}
                    RUNS.append(run)
            if scn == "practical":     # the hop fits are the same in the three scenarios (same snapshots and SNRs)
                for name, h in hops.items():
                    PARAMS.append(dict(n=n, seed=s, fit=name, **h.params(), outside_physics=h.outside_physics(),
                                       at_bound=h.at_bound() if h.bounded else []))
            if n == FIG_N and s == SEEDS[0]:
                KEEP[scn] = models
        print(f"[{scn}] N = {n:3d} done   ({time.time() - t0:5.0f} s)", flush=True)
RUNTIME = time.time() - t0
print(f"Experiment time: {RUNTIME / 60:.1f} min")

# %% [markdown]
# ## 8. Results over seeds
#
# Each number is the mean over the seeds. The seeds change the training snapshots, their noise, and the initial weights of the networks, and the test sets are fixed. Two models are also compared seed by seed on the same snapshots. With 5 seeds, a result obtained in 5 of 5 seeds has a two-sided sign-test $p$-value of 0.0625, so such results are reported as consistent across seeds and not as statistically significant.

# %%
def runs(scn, kind, n=None, arm="bounded"):
    return [x for x in RUNS if x["scenario"] == scn and x["model"] == kind and x["arm"] == arm
            and (n is None or x["n"] == n)]


def stat(scn, kind, n, key, arm="bounded"):
    # mean, standard deviation over seeds, min and max
    v = np.array([x[key] for x in runs(scn, kind, n, arm)])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, float(v.min()), float(v.max())


COLS = [("rmse_inside", "RMSE in"), ("rmse_geometry", "RMSE geo"), ("rmse_storm", "RMSE storm"),
        ("rmse_both", "RMSE both"), ("loss_geometry", "loss geo%"), ("loss_storm", "loss storm%"),
        ("loss_both", "loss both%")]
for scn in SCENARIOS:
    print(f"\n=== {SCN_TITLE[scn]}: mean over {len(SEEDS)} seed(s); RMSE in kbit/s, throughput loss in %")
    print(f"{'model':13s} {'N':>4s} " + " ".join(f"{h:>11s}" for _, h in COLS))
    for kind in KINDS:
        for n in N_TRAIN:
            print(f"{kind:13s} {n:4d} " + " ".join(f"{stat(scn, kind, n, k)[0]:11.2f}" for k, _ in COLS))
    ns = NOMINAL_SCORES[scn]
    print(f"{'Nominal':13s} {'':4s} " + " ".join(f"{ns[k]:11.2f}" for k, _ in COLS))
    for k, v in RULES[scn].items():
        print(f"{k:13s} {'':4s} " + " ".join(f"{v[c]:11.2f}" if c in v else f"{'':11s}" for c, _ in COLS))

RESULTS = dict(mode=MODE, seeds=SEEDS, n_train=N_TRAIN, fig_n=FIG_N, runtime_s=RUNTIME, sanity=SANITY,
               bottleneck=BOTTLENECK, nominal=NOMINAL_SCORES, rules=RULES, runs=RUNS, params=PARAMS,
               overhead_winds=OVH_WINDS,
               bounds=dict(n_s=[1.0, 2.0], zeta=[0.0, None], delta0=[None, None], a=[0.0, None], b=[0.0, None],
                           c=[0.0, None], overhead_coefficients=[0.0, None]),
               test_seeds=dict(main=123, storm=789, both=790, choice_main=456, choice_storm=457, choice_both=458),
               versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                             matplotlib=matplotlib.__version__))
with open(OUT / f"example2_results_{MODE}.json", "w") as fh:
    json.dump(RESULTS, fh, indent=1)
print(f"\nWritten: {OUT / f'example2_results_{MODE}.json'}")


def paired(scn, a, b, n, key, arm_a="bounded", arm_b="bounded"):
    # seed-by-seed difference a - b; wins = seeds in which a is lower (better); two-sided sign-test p-value
    va = {x["seed"]: x[key] for x in runs(scn, a, n, arm_a)}
    vb = {x["seed"]: x[key] for x in runs(scn, b, n, arm_b)}
    d = np.array([va[s] - vb[s] for s in SEEDS])
    k, m = int(np.sum(d < 0)), len(d)
    tail = sum(math.comb(m, i) for i in range(max(k, m - k), m + 1)) / 2 ** m
    return float(d.mean()), float(d.std(ddof=1)) if m > 1 else 0.0, k, min(1.0, 2 * tail)


def paired_line(scn, a, b, n, key, arm_a="bounded", arm_b="bounded", la=None, lb=None):
    d, sd, k, p = paired(scn, a, b, n, key, arm_a, arm_b)
    la, lb = la or LABELS[a], lb or LABELS[b]
    return (f"   {la} minus {lb}, {key}, N={n}: {d:+.3f} ± {sd:.3f} kbit/s; "
            f"first better in {k}/{len(SEEDS)} seeds (sign test p = {p:.3f})")


print("\n=== Calibration audit (hop laws; the same in the three scenarios)")
for n in N_TRAIN:
    line = f"N={n:3d}:"
    for fit in ("chain_unbounded", "coupled_unbounded"):
        P = [p for p in PARAMS if p["fit"] == fit and p["n"] == n]
        bad = sum(1 for p in P if p["outside_physics"])
        names = sorted({k for p in P for k in p["outside_physics"]})
        line += f" | {fit}: {bad}/{len(P)} fits outside physics {names if names else ''}"
    print(line)

print("\n=== Delays (s) at wind " + "/".join(f"{w:g}" for w in OVH_WINDS) + " m/s, practical gateway, mean over seeds; true G1 "
      + "/".join(f"{float(true_switch_overhead_s(0, w)):.2f}" for w in OVH_WINDS) + ", G2 "
      + "/".join(f"{float(true_switch_overhead_s(1, w)):.2f}" for w in OVH_WINDS))
for n in N_TRAIN:
    line = f"   N={n:3d}"
    for kind in ("Coupled+Tq", "Coupled+Tnet"):
        o1 = np.array([x["overhead_G1"] for x in runs("practical", kind, n)]).mean(axis=0)
        o2 = np.array([x["overhead_G2"] for x in runs("practical", kind, n)]).mean(axis=0)
        line += f" | {kind} G1 " + "/".join(f"{v:.2f}" for v in o1) + " G2 " + "/".join(f"{v:.2f}" for v in o2)
    print(line)

print("\n=== Paired comparisons (same snapshots, seed by seed)")
for n in (FIG_N, 500):
    for key in ("rmse_geometry", "rmse_storm", "rmse_both"):
        print(paired_line("practical", "Coupled+Tq", "Coupled+Tnet", n, key))
    for key in ("rmse_storm", "rmse_both"):
        print(paired_line("practical", "Coupled+Tq", "MLP", n, key))
        print(paired_line("practical", "Coupled+Tnet", "MLP", n, key))
    for key in ("rmse_geometry", "rmse_both"):
        print(paired_line("half-duplex", "Coupled+TS", "Coupled+J", n, key))
        print(paired_line("full-duplex", "Coupled+min", "Coupled+J", n, key))
        print(paired_line("practical", "Coupled+Tnet", "HopMLP+Tnet", n, key))
for n in (20, 50, FIG_N):
    for key in ("rmse_storm", "rmse_both"):
        print(paired_line("practical", "Coupled+Tq", "Coupled+Tq", n, key, "bounded", "unbounded",
                          "Coupled+Tq bounded", "Coupled+Tq unbounded"))

# %% [markdown]
# ## 9. Figures
#
# The three gateway behaviors get the same four-panel layout.
#
# - **(a)** End-to-end rate of G1 at 2 km for the models trained on $N = 150$ snapshots (seed 0): versus wind at 60° of elevation for the practical gateway, versus elevation in a 6 m/s wind for the ideal ones. The thin gray lines are the true rates of the two hops, and the shaded band is the measured range.
# - **(b)** RMSE in the three tests at $N = 150$: bars give the mean over seeds, whiskers the best and worst seed.
# - **(c)** RMSE versus $N$ under both shifts (dotted line: nominal chain).
# - **(d)** Throughput lost by the gateway choices at $N = 150$, next to the two single-segment rules (hatched).
#
# `fig_example2_practical.pdf` is Fig. 6 of the article. The figures of the two ideal gateways and the audit of the calibration (`fig_example2_calibration.pdf`) complete it.

# %%
_st = lambda c, m, ls, h=None: dict(color=c, marker=m, ls=ls, hatch=h)
STYLE = {"MLP": _st("#2a78d6", "o", "-"),
         "HopMLP+J": _st("#1baf7a", "^", "-."), "HopMLP+Tnet": _st("#1baf7a", "^", "-."),
         "Chain+min": _st("#eda100", "D", (0, (5, 1.5, 1, 1.5))), "Coupled+min": _st("#eda100", "D", (0, (5, 1.5, 1, 1.5))),
         "Chain+TS": _st("#9b59b6", "v", (0, (1, 1))), "Coupled+TS": _st("#9b59b6", "v", (0, (1, 1))),
         "Chain+J": _st("#eb6834", "s", "--"), "Coupled+J": _st("#eb6834", "s", "--"),
         "Coupled+T0": _st("#8d6e3f", "X", (0, (3, 1, 1, 1))), "Coupled+Tw": _st("#264653", "h", (0, (6, 2))),
         "Coupled+Tq": _st("#b5179e", "P", (0, (2, 1))), "Coupled+Tnet": _st("#eb6834", "s", "--")}
FIG_MODELS = {"practical": ["MLP", "Coupled+TS", "Coupled+Tq", "Coupled+Tnet"],
              "half-duplex": ["MLP", "HopMLP+J", "Coupled+min", "Coupled+TS", "Coupled+J"],
              "full-duplex": ["MLP", "HopMLP+J", "Coupled+min", "Coupled+TS", "Coupled+J"]}
FIG_LABELS = dict(LABELS, **{"Coupled+TS": "Coupled chain + time sharing (no delay)",
                             "Coupled+Tq": "Coupled chain + TS + quadratic delay",
                             "Coupled+Tnet": "Coupled chain + TS + learned delay"})
FIG_FILES = {"practical": "fig_example2_practical.pdf", "half-duplex": "fig_example2_halfduplex.pdf",
             "full-duplex": "fig_example2_fullduplex.pdf"}
RULE_STYLE = {"Nearest buoy": dict(color="#d9d9d6", hatch="////", label="Rule: nearest buoy (acoustic only)"),
              "Always G2": dict(color="#bdbdb8", hatch="\\\\\\\\", label="Rule: always G2 (satellite only)")}


def style_axes(ax, grid_axis="both"):
    ax.grid(True, which="major", axis=grid_axis)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, width=0.6, length=2.5)


def bars(ax, scn, kinds, prefix, rules=False, floor=None):
    # grouped bars over the three tests at N = FIG_N, whiskers from the best to the worst seed
    items = (list(RULE_STYLE) if rules else []) + kinds
    width = 0.8 / len(items)
    for i, item in enumerate(items):
        x = np.arange(len(TESTS)) - 0.4 + (i + 0.5) * width
        if item in RULE_STYLE:
            h = np.array([RULES[scn][item][f"{prefix}_{t}"] for t in TESTS])
            h = np.maximum(h, floor) if floor else h
            ax.bar(x, h, width, color=RULE_STYLE[item]["color"], hatch=RULE_STYLE[item]["hatch"], edgecolor=MUTED,
                   lw=0.4, label=RULE_STYLE[item]["label"])
        else:
            vals = [stat(scn, item, FIG_N, f"{prefix}_{t}") for t in TESTS]
            m = np.array([v[0] for v in vals]); lo = np.array([v[2] for v in vals]); hi = np.array([v[3] for v in vals])
            if floor:
                m, lo, hi = np.maximum(m, floor), np.maximum(lo, floor), np.maximum(hi, floor)
            ax.bar(x, m, width, color=STYLE[item]["color"], lw=0, label=FIG_LABELS[item])
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.0)
    ax.set_xticks(np.arange(len(TESTS))); ax.set_xticklabels(["New\ngeometry", "Storm", "Both\nshifts"])


def lines_vs_n(ax, scn, kinds, key):
    shifts = np.exp(np.linspace(-0.12, 0.12, len(kinds)))          # small horizontal offsets for readability
    for kind, sh in zip(kinds, shifts):
        per = [[x[key] for x in runs(scn, kind, n)] for n in N_TRAIN]
        m = np.array([np.mean(v) for v in per])
        lo, hi = m - np.array([min(v) for v in per]), np.array([max(v) for v in per]) - m
        st = STYLE[kind]
        ax.errorbar(np.array(N_TRAIN) * sh, m, yerr=[lo, hi], color=st["color"], marker=st["marker"], ls=st["ls"],
                    ms=3.0, capsize=1.2, elinewidth=0.5, lw=1.1)
    ax.axhline(NOMINAL_SCORES[scn][key], color=MUTED, ls=":", lw=1.0)
    ax.set_xscale("log"); ax.set_xticks(N_TRAIN); ax.set_xticklabels([str(n) for n in N_TRAIN])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Measured snapshots $N$")


def make_figure(scn, path):
    kinds = FIG_MODELS[scn]
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 4.5))
    # (a) example curves
    ax = axes[0, 0]
    if scn == "practical":                               # versus wind: G1 at 2 km and 60 deg
        x = np.linspace(W_MIN, W_MAX, 200)
        g, r, e, w = np.zeros_like(x, dtype=int), np.full_like(x, 2.0), np.full_like(x, 60.0), x
        ax.axvspan(W_MIN, W_TRAIN, color="#efefec", lw=0)
        ax.set_xlabel("Wind speed (m/s)"); ax.set_xlim(W_MIN, W_MAX); ax.set_title("(a) G1 at 2 km, elevation 60°", pad=3)
    else:                                                # versus elevation: G1 at 2 km, 6 m/s
        x = np.linspace(E_MIN, E_MAX, 200)
        g, r, e, w = np.zeros_like(x, dtype=int), np.full_like(x, 2.0), x, np.full_like(x, 6.0)
        ax.axvspan(E_TRAIN, E_MAX, color="#efefec", lw=0)
        ax.set_xlabel("Satellite elevation (deg)"); ax.set_xlim(E_MIN, E_MAX); ax.set_title("(a) G1 at 2 km, wind 6 m/s", pad=3)
    ra, rs = true_rates(g, r, e, w)
    ax.plot(x, ra, color=LIGHT, lw=0.8); ax.plot(x, rs, color=LIGHT, lw=0.8)
    i_a, i_s = (int(0.72 * len(x)), int(0.03 * len(x))) if scn == "practical" else (int(0.03 * len(x)), int(0.72 * len(x)))
    ax.text(x[i_a], ra[i_a] + 0.4, "acoustic hop", ha="left", va="bottom", fontsize=6, color=MUTED)
    ax.text(x[i_s], rs[i_s] + 0.4, "satellite hop", ha="left", va="bottom", fontsize=6, color=MUTED)
    ax.plot(x, true_e2e(g, r, e, w, scn), color=INK, lw=1.6, label="True end-to-end rate")
    ax.plot(x, NOMINAL.predict(g, r, e, w), color=MUTED, ls=":", lw=1.0, label=LABELS["Nominal"])
    for i, kind in enumerate(kinds):
        st = STYLE[kind]
        ax.plot(x, KEEP[scn][kind].predict(g, r, e, w), color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=2.5, markevery=(i * 6, 30), lw=1.0, label=FIG_LABELS[kind])
    ax.set_ylabel("Rate (kbit/s)"); ax.set_ylim(bottom=0)
    style_axes(ax)
    # (b) RMSE in the three tests
    ax = axes[0, 1]
    bars(ax, scn, kinds, "rmse")
    ax.set_yscale("log"); nice_log(ax.yaxis); ax.set_ylabel("RMSE (kbit/s)"); ax.set_title(f"(b) Prediction error, $N$ = {FIG_N}", pad=3)
    style_axes(ax, "y")
    # (c) RMSE versus N, both shifts
    ax = axes[1, 0]
    lines_vs_n(ax, scn, kinds, "rmse_both")
    ax.set_yscale("log"); nice_log(ax.yaxis); ax.set_ylabel("RMSE (kbit/s)"); ax.set_title("(c) Both shifts at once", pad=3)
    style_axes(ax)
    # (d) gateway choice in the three tests
    ax = axes[1, 1]
    bars(ax, scn, kinds, "loss", rules=True, floor=0.01)
    ax.set_yscale("log"); nice_log(ax.yaxis, subs=(1.0,)); ax.set_ylim(bottom=0.01); ax.set_ylabel("Throughput loss (%)")
    ax.set_title(f"(d) Gateway choice, $N$ = {FIG_N}", pad=3)
    style_axes(ax, "y")
    # one legend for the whole figure
    h, lab = [], []
    for a_ in (axes[0, 0], axes[1, 1]):
        hh, ll = a_.get_legend_handles_labels()
        for hi_, li_ in zip(hh, ll):
            if li_ not in lab:
                h.append(hi_); lab.append(li_)
    fig.tight_layout(w_pad=0.6, h_pad=0.9, rect=(0, 0, 1, 0.81))
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.2,
               columnspacing=0.8)
    fig.savefig(path); plt.show()


for scn in SCENARIOS:
    make_figure(scn, FIG / FIG_FILES[scn])

# audit of the calibration: tilt coefficient and error under both shifts, bounded versus unbounded (practical gateway)
fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9))
ax = axes[0]
for fit, color, mk, dx, lab in (("coupled_unbounded", MUTED, "x", 0.92, "Unbounded"), ("coupled", "#b5179e", "o", 1.08, "Bounded")):
    for n in N_TRAIN:
        c = [p["c_G1"] for p in PARAMS if p["fit"] == fit and p["n"] == n]
        ax.scatter(np.full(len(c), n * dx), c, s=9, color=color, marker=mk, lw=0.8, label=lab if n == N_TRAIN[0] else None)
ax.axhline(TILT_C_SMALL, color=INK, lw=0.8, ls="--"); ax.axhline(0, color=LIGHT, lw=0.6)
ax.text(N_TRAIN[-1] * 1.2, TILT_C_SMALL, " small-tilt\n value", fontsize=6, va="center", color=INK)
ax.set_xscale("log"); ax.set_xticks(N_TRAIN); ax.set_xticklabels([str(n) for n in N_TRAIN])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("Measured snapshots $N$"); ax.set_ylabel("Fitted tilt coefficient $c$"); ax.set_title("(a) One fit per seed", pad=3)
ax.legend(frameon=False, loc="lower right"); style_axes(ax)
ax = axes[1]
for arm, color, ls, lab in (("unbounded", MUTED, "--", "Unbounded"), ("bounded", "#b5179e", "-", "Bounded")):
    m = [stat("practical", "Coupled+Tq", n, "rmse_both", arm)[0] for n in N_TRAIN]
    ax.plot(N_TRAIN, m, color=color, ls=ls, marker="o", ms=3, label=lab)
ax.set_xscale("log"); ax.set_ylim(bottom=0); ax.set_xticks(N_TRAIN); ax.set_xticklabels([str(n) for n in N_TRAIN])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("Measured snapshots $N$"); ax.set_ylabel("RMSE, both shifts (kbit/s)")
ax.set_title("(b) Coupled chain + TS + quadratic delay", pad=3); ax.legend(frameon=False); style_axes(ax)
fig.tight_layout(w_pad=0.8)
fig.savefig(FIG / "fig_example2_calibration.pdf"); plt.show()
print("Written:", ", ".join(str(FIG / f) for f in list(FIG_FILES.values()) + ["fig_example2_calibration.pdf",
                                                                              "fig_example2_setup.pdf"]))

# %% [markdown]
# ## 10. Reading the results
#
# 1. **One hop is not enough.** Choosing the buoy from the acoustic hop alone (nearest buoy) or from the satellite hop alone (always G2) loses several percent of the throughput in the new geometry and under both shifts, whereas the coupled physical models lose almost nothing.
# 2. **A simple rule can win where one gateway dominates.** In the storm, G2 is almost always the better buoy, so *always G2* is nearly optimal.
# 3. **The laws carry the extrapolation.** The hop MLPs receive the same per-hop measurements as the calibrated laws, yet they extrapolate much worse.
# 4. **The largest error is at the interface.** Ignoring the switching delay of the practical buoy costs more than any choice of its form, by far in the storm.
# 5. **A known interface needs no learning.** For the full-duplex and ideal half-duplex buoys, the matching formula matches or beats the learned junction.
# 6. **For an unknown interface, a simple form chosen from physical reasoning comes first.** Inside the measured winds, all delay forms are close. Beyond them, a $\tanh$ network flattens, whereas a parabola keeps growing like the true delay.
# 7. **Physics also bounds the calibration.** With 20 snapshots, plain least squares can return a negative tilt loss or an impossible spreading factor.
#
# **Limits.** The measurements are generated by this code. Distances, elevations, and winds are drawn independently, so satellite passes, buoy trajectories, and buffering are not simulated. The hops are coupled through the wind only. Per-hop SNRs and the throughput carry independent random errors, but no bias, outage, or missing data. With five seeds, differences are reported as consistent or not across seeds.

# %% [markdown]
# ## 11. Try it yourself
#
# `W_TRAIN_TRY` is the strongest wind that was measured (8 m/s in the article). Raising it toward 14 m/s gives the models measurements in rough seas. The cell uses the practical gateway, $N = 150$, and seed 0 (about 30 s).

# %%
W_TRAIN_TRY = 11.0          # strongest measured wind in m/s (article: 8.0)

rng = np.random.default_rng(0)
g, r, eps, wind = sample_snapshots(150, rng, w_hi=W_TRAIN_TRY)
snr_a, snr_s, y = measure(g, r, eps, wind, rng, "practical")
models, _, _ = build_models(dict(g=g, r=r, eps=eps, wind=wind, snr_a=snr_a, snr_s=snr_s, y=y), 0)
print(f"Measured winds up to {W_TRAIN_TRY:g} m/s, practical gateway, N = 150, seed 0. RMSE in kbit/s:")
print(f"{'model':46s} {'storm':>8s} {'both shifts':>12s}")
for kind in ["MLP", "HopMLP+Tnet", "Coupled+TS", "Coupled+Tw", "Coupled+Tq", "Coupled+Tnet"]:
    print(f"{LABELS[kind]:46s} {rmse(models[kind].predict(*STORM), TRUTH['practical']['storm']):8.2f} "
          f"{rmse(models[kind].predict(*BOTH), TRUTH['practical']['both']):12.2f}")
