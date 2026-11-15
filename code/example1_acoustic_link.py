# %% [markdown]
# # Worked example 1: a physics-informed model of an underwater acoustic link
#
# Companion code of *Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey* (M. Jouhari, H. Benaddi, M. Guizani), **Section II-F**. Repository: https://github.com/mohammed-jouhari/PIML-SAGSIN-Survey
#
# **The question.** The loss of an underwater acoustic link is measured at short range and low frequency only. Which model predicts the loss where nothing was measured, and which one indicates when it should not be trusted?
#
# **The models.** Every model receives the same noisy measurements.
#
# | Model | What it knows | Mechanism code (Table II of the article) |
# |---|---|---|
# | **MLP** | only data | none |
# | **GP, constant mean** | only data, with error bars | none |
# | **PINN** | the form of the law, with $n_s$ and $\zeta$ learned | loss term (L) |
# | **Hybrid, textbook law** | the law with textbook parameters, plus a learned correction | hybrid residual (R) |
# | **Hybrid, joint** | the law with parameters learned together with the correction | R |
# | **Hybrid, calibrated law** | the law calibrated first by least squares, then a learned correction | R |
# | **Calibrated law** | the form of the law, three parameters fitted by least squares | the law itself |
# | **GP + law** (universal kriging) | the law as the mean, with a GP correction that fades away from the data | R, probabilistic |
#
# **Two oceans.** In ocean **A**, the law has the right form but wrong textbook parameters. In ocean **B**, the law has the wrong form: a rough sea surface reflects sound and changes how the loss grows with range, which no model contains.
#
# **Run time** on a free Colab CPU: `MODE = "full"` (5 seeds) takes about 40 minutes and `MODE = "quick"` (1 seed) about 10 minutes. Only PyTorch, NumPy, SciPy, and Matplotlib are used. From a terminal, `NB_MODE=quick python example1_acoustic_link.py` runs the quick mode.

# %%
import os, json, time, math
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
import torch
import matplotlib
if not os.environ.get("DISPLAY") and "ipykernel" not in __import__("sys").modules:
    matplotlib.use("Agg")                        # no window when run as a script on a machine without display
import matplotlib.pyplot as plt

# "full": the 5 seeds of the article (about 40 min on a free Colab CPU). "quick": 1 seed (about 10 min).
MODE = "full"
MODE = os.environ.get("NB_MODE", MODE)           # the environment variable NB_MODE overrides the line above
SEEDS = [0] if MODE == "quick" else [0, 1, 2, 3, 4]
N_TRAIN = [10, 30, 100, 300]                     # numbers of measurements

torch.set_num_threads(2)                         # the article used 2 CPU threads, the size of a free Colab machine
OUT = Path("results"); OUT.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
print(f"MODE = {MODE}, seeds = {SEEDS}")
print(f"torch {torch.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}")

# %% [markdown]
# **Reproducibility across machines.** The numbers of the article come from a full run on a free Colab CPU. On another CPU, all models give the same errors to within 0.003 dB, except two: the PINN, whose L-BFGS stage amplifies floating-point rounding (its mean errors can move by up to about 20%), and, for single seeds, the calibrated hybrid, whose network fits the noise. The check of Section 10 may then flag a few PINN rows, and the conclusions do not change.

# %% [markdown]
# ## 1. The two oceans
#
# The **Thorp formula** gives the absorption of sound in sea water, with $f$ in kHz (Eq. (13) of the article):
# $$\alpha(f) = \frac{0.11 f^2}{1+f^2} + \frac{44 f^2}{4100+f^2} + 2.75\cdot 10^{-4} f^2 + 0.003 \quad \text{dB/km}.$$
# The law used by every physical model is spreading plus absorption plus an offset (Eq. (5)):
# $$\mathrm{TL}(l,f) = 10\, n_s \log_{10}\!\Big(\frac{l}{l_{\mathrm{ref}}}\Big) + \zeta\, l\, \alpha(f) + \Delta_0, \qquad l_{\mathrm{ref}} = 1~\text{m}.$$
# The textbook values are $n_s = 1.5$, $\zeta = 1$, $\Delta_0 = 0$.
#
# **Ocean A (right form, wrong parameters).** The true values are $n_s = 1.7$, $\zeta = 1.25$, $\Delta_0 = 3$ dB, and a ripple $\rho(l) = 2\sin(2\pi l/1.5)\,e^{-l/4}$ dB mimics interference. No model contains the ripple.
#
# **Ocean B (wrong form).** The two nodes are 10 m below a rough sea surface. Sound reaches the receiver directly and after a reflection on the surface, and the two paths interfere (Lloyd's mirror; Jensen et al., *Computational Ocean Acoustics*, 2011). The roughness keeps only part of the reflection coherent, with the coefficient $\mu = \exp\!\big(-2 (k\,\sigma_h \sin\theta)^2\big)$, where $k = 2\pi f / c$, $\sigma_h = 0.5$ m is the rms wave height, and $\sin\theta = (z_s + z_r)/l$. The intensity relative to the direct path is
# $$G(l,f) = 2\big(1 - \mu \cos(k\,\Delta R)\big), \qquad \Delta R = \frac{2 z_s z_r}{l},$$
# and the true TL is the law with the true parameters plus $-10\log_{10} G$. Far from the source and at low frequency, the loss grows as $40\log_{10} l$, faster than any $n_s \le 2$ allows.
#
# Both oceans are simulations with known truth, so that every error can be measured exactly.

# %%
L_MIN, L_MAX, F_MIN, F_MAX = 0.2, 10.0, 2.0, 40.0     # full domain: range in km, frequency in kHz
L_TRAIN, F_TRAIN = 5.0, 25.0                          # measured region: l <= 5 km and f <= 25 kHz
TRUE = dict(k=1.7, gamma=1.25, c0=3.0)                # true parameters (k = n_s, gamma = zeta, c0 = Delta_0)
NOMINAL = dict(k=1.5, gamma=1.0, c0=0.0)              # textbook values
RIPPLE_DB, RIPPLE_PERIOD_KM, RIPPLE_DECAY_KM = 2.0, 1.5, 4.0
NOISE_DB = 1.5                                        # measurement noise (standard deviation, dB)
Z_SRC_M, Z_RCV_M, WAVE_RMS_M, C_SOUND = 10.0, 10.0, 0.5, 1500.0   # ocean B: depths, rms wave height, sound speed


def thorp_db_per_km(f_khz):
    # Thorp absorption in dB/km, f in kHz
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def transmission_loss_db(l_km, f_khz, k=1.5, gamma=1.0, c0=0.0, l_ref_km=1e-3):
    # the law: 10 n_s log10(l / l_ref) + zeta l alpha(f) + Delta_0
    l = np.asarray(l_km, dtype=float)
    return 10 * k * np.log10(l / l_ref_km) + gamma * l * thorp_db_per_km(f_khz) + c0


def ripple_db(l_km):
    return RIPPLE_DB * np.sin(2 * np.pi * np.asarray(l_km) / RIPPLE_PERIOD_KM) * np.exp(-np.asarray(l_km) / RIPPLE_DECAY_KM)


def surface_interference_db(l_km, f_khz):
    # ocean B: extra loss -10 log10(G) of the surface-reflected path (Lloyd's mirror under a rough surface)
    l = np.asarray(l_km, dtype=float) * 1e3
    k = 2 * np.pi * np.asarray(f_khz, dtype=float) * 1e3 / C_SOUND
    dR, sin_t = 2 * Z_SRC_M * Z_RCV_M / l, (Z_SRC_M + Z_RCV_M) / l
    mu = np.exp(-2 * (k * WAVE_RMS_M * sin_t) ** 2)
    return -10 * np.log10(2 * (1 - mu * np.cos(k * dR)))


def true_tl_A(l, f):
    # ocean A: the law with the true parameters, plus a ripple that no model has
    return transmission_loss_db(l, f, **TRUE) + ripple_db(l)


def true_tl_B(l, f):
    # ocean B: the law with the true parameters, plus the surface interference that no model has
    return transmission_loss_db(l, f, **TRUE) + surface_interference_db(l, f)


OCEANS = {"A": true_tl_A, "B": true_tl_B}
OCEAN_TITLE = {"A": "ocean A (right form of the law)", "B": "ocean B (wrong form of the law)"}

for fq in (10, 20, 50):
    print(f"Thorp absorption at {fq:2d} kHz: {thorp_db_per_km(fq):5.1f} dB/km")
_L, _F = np.meshgrid(np.linspace(L_MIN, L_MAX, 300), np.linspace(F_MIN, F_MAX, 300))
_E = surface_interference_db(_L, _F)
print(f"Ocean B: surface interference between {_E.min():.1f} and {_E.max():.1f} dB over the full domain")

# %% [markdown]
# ## 2. Measurements and test points
#
# Training samples are drawn uniformly in the measured region ($0.2 \le l \le 5$ km, $2 \le f \le 25$ kHz) with Gaussian noise of 1.5 dB. The test points are 4000 random points of the full domain, compared with the noise-free truth. The outside points are also split by the kind of shift.
#
# | Test set | Range | Frequency |
# |---|---|---|
# | inside | $\le 5$ km | $\le 25$ kHz |
# | outside | all of the rest | |
# | farther | $> 5$ km | $\le 25$ kHz |
# | higher frequency | $\le 5$ km | $> 25$ kHz |
# | both | $> 5$ km | $> 25$ kHz |
#
# Two references need no learning: the textbook law alone and, for ocean A, the exact law (true parameters, no ripple), whose error is the floor of any model without the ripple.

# %%
def sample_train(n, rng, truth):
    l = rng.uniform(L_MIN, L_TRAIN, n)
    f = rng.uniform(F_MIN, F_TRAIN, n)
    return l, f, truth(l, f) + rng.normal(0, NOISE_DB, n)


def test_sets(rng, n=4000):
    l, f = rng.uniform(L_MIN, L_MAX, n), rng.uniform(F_MIN, F_MAX, n)
    near, low = l <= L_TRAIN, f <= F_TRAIN
    return {"inside": (l[near & low], f[near & low]), "outside": (l[~(near & low)], f[~(near & low)]),
            "farther": (l[~near & low], f[~near & low]), "higher_f": (l[near & ~low], f[near & ~low]),
            "both": (l[~near & ~low], f[~near & ~low])}


TESTS = test_sets(np.random.default_rng(123))
REGIONS = list(TESTS)
print({k: len(v[0]) for k, v in TESTS.items()})

rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
TEXTBOOK_RMSE = {o: {k: rms(transmission_loss_db(*v, **NOMINAL) - OCEANS[o](*v)) for k, v in TESTS.items()} for o in OCEANS}
ORACLE_RMSE_A = {k: rms(transmission_loss_db(*v, **TRUE) - true_tl_A(*v)) for k, v in TESTS.items()}
for o in OCEANS:
    print(f"Textbook law alone, {OCEAN_TITLE[o]}: " + ", ".join(f"{k} {v:.1f}" for k, v in TEXTBOOK_RMSE[o].items()) + " dB")
print("Exact law (true parameters, no ripple), ocean A: " + ", ".join(f"{k} {v:.2f}" for k, v in ORACLE_RMSE_A.items()) + " dB")

# %% [markdown]
# ## 3. Networks: MLP and PINN
#
# All networks share one body: two hidden layers of 64 units, $\tanh$ activations, inputs $\ln l$ and $f$ scaled to $[0, 1]$. The **MLP** is trained on the data only with Adam (3000 steps, learning rate $2\cdot10^{-3}$, cosine decay).
#
# The **PINN** adds the residuals of the law (Eq. (6) of the article) at 1000 collocation points spread over the full domain, which need no labels:
# $$r_l = \frac{\partial \widehat{\mathrm{TL}}}{\partial \ln l} - \frac{10\, n_s}{\ln 10} - \zeta\, l\, \alpha(f), \qquad r_f = \frac{\partial \widehat{\mathrm{TL}}}{\partial f} - \zeta\, l\, \frac{\mathrm{d}\alpha}{\mathrm{d}f}.$$
# $n_s$ and $\zeta$ are learned with the weights, starting from the textbook values. Training: 2000 Adam steps, then up to 400 L-BFGS iterations.

# %%
STEPS, LR, TL_SCALE = 3000, 2e-3, 100.0
N_COLLOC, LAMBDA, PINN_ADAM_STEPS, PINN_LBFGS_ITER = 1000, 1.0, 2000, 400


def features(l, f):
    return torch.stack([(torch.log(l) - np.log(L_MIN)) / (np.log(L_MAX) - np.log(L_MIN)), f / F_MAX], dim=-1)


class Net(torch.nn.Module):
    # 2 inputs -> 64 -> 64 -> 1, tanh. Shared by the MLP, the PINN and the hybrid models.
    def __init__(self, width=64):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(),
                                        torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))

    def forward(self, l, f):
        return self.body(features(l, f)).squeeze(-1) * TL_SCALE


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = Net()

    def forward(self, l, f):
        return self.net(l, f)


def adam_loop(model, loss_fn, steps):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step(); sched.step()


def to_t(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


def fit_mlp(l, f, y, seed):
    torch.manual_seed(seed); model = MLP(); lt, ft, yt = to_t(l, f, y)
    adam_loop(model, lambda: torch.mean(((model(lt, ft) - yt) / 10.0) ** 2), STEPS)
    return model


def thorp_t(f_khz):
    f2 = f_khz ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


class PINN(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = Net()
        self.k = torch.nn.Parameter(torch.tensor(NOMINAL["k"]))          # n_s, learned
        self.gamma = torch.nn.Parameter(torch.tensor(NOMINAL["gamma"]))  # zeta, learned

    def forward(self, l, f):
        return self.net(l, f)

    def residual(self, l, f):
        l = l.clone().requires_grad_(True); f = f.clone().requires_grad_(True)
        tl = self.net(l, f)
        dtl_dl, dtl_df = torch.autograd.grad(tl.sum(), (l, f), create_graph=True)     # exact derivatives
        alpha = thorp_t(f); dalpha_df = torch.autograd.grad(alpha.sum(), f, create_graph=True)[0]
        r_x = l * dtl_dl - (10 * self.k / np.log(10) + self.gamma * l * alpha)
        r_f = dtl_df - self.gamma * l * dalpha_df
        return torch.cat([r_x, r_f])


def fit_pinn(l, f, y, seed):
    torch.manual_seed(seed); model = PINN(); lt, ft, yt = to_t(l, f, y)
    g = torch.Generator().manual_seed(1000 + seed)
    lc = L_MIN + (L_MAX - L_MIN) * torch.rand(N_COLLOC, generator=g)     # collocation over the full domain
    fc = F_MIN + (F_MAX - F_MIN) * torch.rand(N_COLLOC, generator=g)

    def loss_fn():
        return torch.mean(((model(lt, ft) - yt) / 10.0) ** 2) + LAMBDA * torch.mean((model.residual(lc, fc) / 10.0) ** 2)
    adam_loop(model, loss_fn, PINN_ADAM_STEPS)
    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=PINN_LBFGS_ITER, history_size=50,
                           line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        lb.zero_grad(); loss = loss_fn(); loss.backward(); return loss
    lb.step(closure)
    return model

# %% [markdown]
# ## 4. Calibrated law and hybrid models
#
# The law is linear in $(n_s, \zeta, \Delta_0)$, so the **calibrated law** fits them by ordinary least squares, with no network.
#
# The **hybrid residual** (mechanism R, Eq. (1) of the article) predicts $\hat y = f_{\mathrm{phys}}(l,f) + g_\theta(l,f)$. The physical part is given three ways, with the same network and training as the MLP:
#
# - **Hybrid, textbook law**: $n_s = 1.5$, $\zeta = 1$, $\Delta_0 = 0$.
# - **Hybrid, joint**: $n_s$, $\zeta$, $\Delta_0$ are trained together with the network, starting from the textbook values.
# - **Hybrid, calibrated law**: the law is first calibrated by least squares, then the network learns what the calibrated law misses.

# %%
class Calibrated:
    # the law with (n_s, zeta, Delta_0) fitted by linear least squares; no network
    def __init__(self, l, f, y):
        A = np.stack([10 * np.log10(l / 1e-3), l * thorp_db_per_km(f), np.ones_like(l)], axis=1)
        (self.k, self.gamma, self.c0), *_ = np.linalg.lstsq(A, y, rcond=None)

    def predict(self, l, f):
        return transmission_loss_db(l, f, k=self.k, gamma=self.gamma, c0=self.c0)


class Hybrid(torch.nn.Module):
    # f_phys(l, f) + g_theta(l, f); law = "textbook", "joint" (learned with the network) or "calibrated" (LS first)
    def __init__(self, law="textbook", cal=None):
        super().__init__(); self.net = Net(); self.law = law
        if law == "joint":
            self.k, self.gamma, self.c0 = (torch.nn.Parameter(torch.tensor(float(NOMINAL[x]))) for x in ("k", "gamma", "c0"))
        else:
            src = NOMINAL if law == "textbook" else dict(k=cal.k, gamma=cal.gamma, c0=cal.c0)
            self.k, self.gamma, self.c0 = (torch.tensor(float(src[x])) for x in ("k", "gamma", "c0"))

    def forward(self, l, f):
        phys = 10 * self.k * torch.log10(l / 1e-3) + self.gamma * l * thorp_t(f) + self.c0
        return phys + self.net(l, f) / 10.0          # the correction works at a finer scale (units of 10 dB)


def fit_hybrid(l, f, y, seed, law):
    torch.manual_seed(seed)
    model = Hybrid(law, Calibrated(l, f, y) if law == "calibrated" else None)
    lt, ft, yt = to_t(l, f, y)
    adam_loop(model, lambda: torch.mean(((model(lt, ft) - yt) / 10.0) ** 2), STEPS)
    return model

# %% [markdown]
# ## 5. Gaussian processes: universal kriging with the law as mean
#
# A Gaussian process (GP) predicts a mean plus a correction that follows the data where they exist and returns to the mean far from them, with a standard deviation. The mean is $h(\mathbf{x})^\top\boldsymbol\beta$, with two choices of basis functions:
#
# - **GP + law**: $h = [\,10\log_{10}(l/l_{\mathrm{ref}}),\ l\,\alpha(f),\ 1\,]$, so $\boldsymbol\beta = (n_s, \zeta, \Delta_0)$ is the calibrated law.
# - **GP, constant mean**: $h = [1]$, a data-only GP.
#
# $\boldsymbol\beta$ is estimated jointly with the GP by generalized least squares (universal kriging; Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006, Sec. 2.7), and the predictive variance includes the uncertainty of $\boldsymbol\beta$. The kernel is a squared exponential with one length scale per input plus Gaussian noise, and its four hyperparameters maximize the restricted likelihood from 6 starting points.
#
# **One physical constraint.** With few noisy samples, a GP with a very long length scale can imitate the smooth trend of the law, and $\boldsymbol\beta$ then drifts away from the physical values. The range length scale is therefore bounded by half of the measured range (2.5 km), so that the GP only models local departures from the law. Section 9 shows the effect of this bound.

# %%
GP_LS_RANGE_MAX = 0.25        # range length scale <= 0.25 x 10 km = 2.5 km, half of the measured range
GP_BOUNDS = [(np.log(0.05), np.log(30.0)),             # signal std (dB)
             (np.log(0.02), np.log(GP_LS_RANGE_MAX)),  # length scale in l / 10 km
             (np.log(0.02), np.log(5.0)),              # length scale in f / 40 kHz
             (np.log(0.3), np.log(8.0))]               # noise std (dB)


def basis(l, f, mean):
    l, f = np.asarray(l, dtype=float), np.asarray(f, dtype=float)
    if mean == "law":
        return np.stack([10 * np.log10(l / 1e-3), l * thorp_db_per_km(f), np.ones_like(l)], axis=1)
    return np.ones((len(l), 1))


def gp_inputs(l, f):
    return np.stack([np.asarray(l, dtype=float) / L_MAX, np.asarray(f, dtype=float) / F_MAX], axis=1)


def sq_exp(X1, X2, sf, ls):
    d = (X1[:, None, :] - X2[None, :, :]) / ls
    return sf ** 2 * np.exp(-0.5 * np.sum(d ** 2, axis=2))


class KrigingGP:
    # Universal kriging: y = h(x)^T beta + GP(0, k) + noise; beta by GLS; hyperparameters by REML.
    def __init__(self, l, f, y, mean="law", seed=0, bounds=None):
        self.mean, self.bounds = mean, bounds or GP_BOUNDS
        self.X, self.H, self.y = gp_inputs(l, f), basis(l, f, mean), np.asarray(y, dtype=float)
        rng = np.random.default_rng(seed)
        starts = [np.array([np.log(2.0), np.log(0.1), np.log(1.0), np.log(1.5)])]
        starts += [np.array([rng.uniform(*b) for b in self.bounds]) for _ in range(5)]
        x0 = [np.clip(s, [b[0] for b in self.bounds], [b[1] for b in self.bounds]) for s in starts]
        fits = [minimize(self._neg_reml, s, method="L-BFGS-B", bounds=self.bounds) for s in x0]
        self._set(min(fits, key=lambda r: r.fun).x)

    def _chol(self, th):
        sf, l1, l2, sn = np.exp(th)
        K = sq_exp(self.X, self.X, sf, np.array([l1, l2])) + (sn ** 2 + 1e-8) * np.eye(len(self.y))
        return np.linalg.cholesky(K)

    def _neg_reml(self, th):
        try:
            L = self._chol(th)
        except np.linalg.LinAlgError:
            return 1e10
        solve = lambda B: np.linalg.solve(L.T, np.linalg.solve(L, B))
        A = self.H.T @ solve(self.H)
        try:
            LA = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return 1e10
        beta = np.linalg.solve(A, self.H.T @ solve(self.y))
        r = self.y - self.H @ beta
        n, p = self.H.shape
        return 0.5 * r @ solve(r) + np.log(np.diag(L)).sum() + np.log(np.diag(LA)).sum() + 0.5 * (n - p) * np.log(2 * np.pi)

    def _set(self, th):
        self.theta = th
        self.sf, self.ls, self.sn = np.exp(th[0]), np.exp(th[1:3]), np.exp(th[3])
        self.L = self._chol(th)
        solve = lambda B: np.linalg.solve(self.L.T, np.linalg.solve(self.L, B))
        self.A = self.H.T @ solve(self.H)
        self.beta = np.linalg.solve(self.A, self.H.T @ solve(self.y))
        self.w = solve(self.y - self.H @ self.beta)
        if self.mean == "law":
            self.k, self.gamma, self.c0 = self.beta

    def predict(self, l, f, return_std=False):
        Xs, Hs = gp_inputs(l, f), basis(l, f, self.mean)
        Ks = sq_exp(Xs, self.X, self.sf, self.ls)
        mu = Hs @ self.beta + Ks @ self.w
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, Ks.T)
        var = self.sf ** 2 - np.sum(v ** 2, axis=0)
        R = Hs.T - self.H.T @ np.linalg.solve(self.L.T, v)        # h* - H^T K^-1 k*
        var = var + np.sum(R * np.linalg.solve(self.A, R), axis=0)  # + uncertainty of beta
        return mu, np.sqrt(np.maximum(var, 1e-12))                  # std of the noise-free TL


FITTERS = {
    "MLP": fit_mlp,
    "GP-const": lambda l, f, y, s: KrigingGP(l, f, y, "const", s),
    "PINN": fit_pinn,
    "Hybrid-textbook": lambda l, f, y, s: fit_hybrid(l, f, y, s, "textbook"),
    "Hybrid-joint": lambda l, f, y, s: fit_hybrid(l, f, y, s, "joint"),
    "Hybrid-calibrated": lambda l, f, y, s: fit_hybrid(l, f, y, s, "calibrated"),
    "Calibrated": lambda l, f, y, s: Calibrated(l, f, y),
    "GP+law": lambda l, f, y, s: KrigingGP(l, f, y, "law", s),
}
LABELS = {"MLP": "MLP (data only)", "GP-const": "GP, constant mean (data only)", "PINN": "PINN (physics in the loss)",
          "Hybrid-textbook": "Hybrid R, textbook law", "Hybrid-joint": "Hybrid R, joint calibration",
          "Hybrid-calibrated": "Hybrid R, calibrated law", "Calibrated": "Calibrated law (LS)",
          "GP+law": "GP + calibrated law (kriging)"}


def predict(model, l, f):
    if isinstance(model, (Calibrated, KrigingGP)):
        return model.predict(l, f)
    with torch.no_grad():
        return model(*to_t(l, f)).numpy()


def law_params(model):
    if hasattr(model, "k"):
        return tuple(float(v.detach() if torch.is_tensor(v) else v) for v in (model.k, model.gamma))
    return None

# %% [markdown]
# ## 6. Check of the pipeline
#
# 1. The PINN on 300 samples without noise and without the ripple over the full domain must recover $n_s = 1.7$ and $\zeta = 1.25$.
# 2. The GP + law on the same clean data must recover $(n_s, \zeta, \Delta_0) = (1.7, 1.25, 3)$.
# 3. On ocean A with 300 noisy samples, the 95% intervals of the GP + law should contain about 95% of the inside test points.

# %%
t0 = time.time()
_rng = np.random.default_rng(0)
_l, _f = _rng.uniform(L_MIN, L_MAX, 300), _rng.uniform(F_MIN, F_MAX, 300)
_y = transmission_loss_db(_l, _f, **TRUE)
SANITY = dict(zip(("pinn_k", "pinn_gamma"), law_params(fit_pinn(_l, _f, _y, 0))))
_g = KrigingGP(_l, _f, _y, "law", 0)
SANITY.update(gp_k=float(_g.k), gp_gamma=float(_g.gamma), gp_c0=float(_g.c0))
print(f"PINN on clean data:     n_s = {SANITY['pinn_k']:.3f}, zeta = {SANITY['pinn_gamma']:.3f}   (true 1.7, 1.25)")
print(f"GP + law on clean data: n_s = {SANITY['gp_k']:.3f}, zeta = {SANITY['gp_gamma']:.3f}, Delta_0 = {SANITY['gp_c0']:.2f}"
      "   (true 1.7, 1.25, 3)")
_l, _f, _y = sample_train(300, np.random.default_rng(0), true_tl_A)
_g = KrigingGP(_l, _f, _y, "law", 0)
_m, _s = _g.predict(*TESTS["inside"], return_std=True)
SANITY["gp_cover_inside_A300"] = float(np.mean(np.abs(_m - true_tl_A(*TESTS["inside"])) <= 1.96 * _s))
print(f"GP + law, ocean A, N = 300, seed 0: 95% coverage inside = {100 * SANITY['gp_cover_inside_A300']:.1f}%"
      f"   [{time.time() - t0:.0f} s]")

# %% [markdown]
# ## 7. Experiment
#
# For each ocean, number of samples $N$, and seed, the eight models are trained on the same samples and tested on the same points. In `full` mode this is $2 \times 4 \times 5 = 40$ training sets and 320 fits.

# %%
def evaluate(model, truth):
    out = {f"rmse_{k}": rms(predict(model, *v) - truth(*v)) for k, v in TESTS.items()}
    if isinstance(model, KrigingGP):
        for k, v in TESTS.items():
            m, s = model.predict(*v, return_std=True)
            out[f"cover_{k}"] = float(np.mean(np.abs(m - truth(*v)) <= 1.96 * s))
            out[f"width_{k}"] = float(np.mean(2 * 1.96 * s))
        out.update(gp_sf=float(model.sf), gp_ls_l_km=float(model.ls[0] * L_MAX), gp_ls_f_khz=float(model.ls[1] * F_MAX),
                   gp_sn=float(model.sn))
    p = law_params(model)
    if p is not None:
        out["k"], out["gamma"] = p
    return out


t0 = time.time()
RUNS, KEEP = [], {}
for ocean, truth in OCEANS.items():
    for n in N_TRAIN:
        for s in SEEDS:
            l, f, y = sample_train(n, np.random.default_rng(s), truth)
            for kind, fit in FITTERS.items():
                t1 = time.time()
                m = fit(l, f, y, s)
                RUNS.append(dict(ocean=ocean, model=kind, n=n, seed=s, fit_s=time.time() - t1, **evaluate(m, truth)))
                if n == 100 and s == SEEDS[0]:                      # kept for the curve panels of the figure
                    KEEP[(ocean, kind)] = m
        print(f"[ocean {ocean}] N = {n:3d} done   ({time.time() - t0:5.0f} s)", flush=True)
RUNTIME = time.time() - t0
print(f"Experiment time: {RUNTIME / 60:.1f} min")

# %% [markdown]
# ## 8. Results over seeds
#
# Each cell is the mean over seeds. The seeds change the training samples, their noise, and the initial weights, and the test points are fixed. Two models are also compared seed by seed on the same samples. With 5 seeds, a result obtained in 5 of 5 seeds has a two-sided sign-test $p$-value of 0.0625, so such results are reported as consistent across seeds and not as statistically significant.

# %%
KINDS = list(FITTERS)


def runs(ocean, kind, n=None):
    return [r for r in RUNS if r["ocean"] == ocean and r["model"] == kind and (n is None or r["n"] == n)]


def stat(ocean, kind, n, key):
    v = np.array([r[key] for r in runs(ocean, kind, n)])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, float(v.min()), float(v.max())


for ocean in OCEANS:
    print(f"\n=== {OCEAN_TITLE[ocean]}: mean RMSE over {len(SEEDS)} seed(s), dB")
    print(f"{'model':34s} {'N':>4s} " + " ".join(f"{k:>9s}" for k in REGIONS))
    for kind in KINDS:
        for n in N_TRAIN:
            print(f"{LABELS[kind]:34s} {n:4d} " + " ".join(f"{stat(ocean, kind, n, 'rmse_' + k)[0]:9.2f}" for k in REGIONS))
    print(f"{'Textbook law alone':34s} {'':4s} " + " ".join(f"{TEXTBOOK_RMSE[ocean][k]:9.2f}" for k in REGIONS))
    if ocean == "A":
        print(f"{'Exact law (no ripple)':34s} {'':4s} " + " ".join(f"{ORACLE_RMSE_A[k]:9.2f}" for k in REGIONS))

for ocean in OCEANS:
    print(f"\n=== {OCEAN_TITLE[ocean]}: 95% coverage (%) and mean width (dB) of the GP intervals, mean over seeds")
    for kind in ("GP+law", "GP-const"):
        for n in N_TRAIN:
            print(f"   {LABELS[kind]:32s} N = {n:3d}: " + " | ".join(
                f"{k} {100 * stat(ocean, kind, n, 'cover_' + k)[0]:5.1f}% ({stat(ocean, kind, n, 'width_' + k)[0]:5.1f} dB)"
                for k in ("inside", "farther", "higher_f", "both")))

print("\n=== Mean fitting time per model (s), all fits")
print(", ".join(f"{LABELS[k]} {np.mean([r['fit_s'] for r in RUNS if r['model'] == k]):.1f}" for k in KINDS))


def paired(ocean, a, b, n, key):
    va = {r["seed"]: r[key] for r in runs(ocean, a, n)}
    vb = {r["seed"]: r[key] for r in runs(ocean, b, n)}
    d = np.array([va[s] - vb[s] for s in SEEDS])
    k, m = int(np.sum(d < 0)), len(d)
    tail = sum(math.comb(m, i) for i in range(max(k, m - k), m + 1)) / 2 ** m
    return float(d.mean()), k, min(1.0, 2 * tail)


PAIRS = [("A", "Hybrid-calibrated", "Hybrid-textbook", "outside", "Q1. Does calibrating the physics of the hybrid fix its extrapolation?"),
         ("A", "Hybrid-joint", "Hybrid-textbook", "outside", ""),
         ("A", "Hybrid-calibrated", "PINN", "outside", "Q2. With calibrated physics on both sides, R against L"),
         ("A", "Hybrid-calibrated", "Calibrated", "inside", "Q3. Does the network correction help the calibrated law?"),
         ("A", "Hybrid-calibrated", "Calibrated", "outside", ""),
         ("A", "GP+law", "Calibrated", "inside", "Q4. Does the GP + law learn the ripple and keep the law outside?"),
         ("A", "GP+law", "Calibrated", "outside", ""),
         ("A", "GP+law", "Hybrid-calibrated", "outside", ""),
         ("B", "Hybrid-calibrated", "Calibrated", "inside", "Q5. Wrong form: does a learned correction beat the law?"),
         ("B", "GP+law", "Calibrated", "inside", ""),
         ("B", "Hybrid-calibrated", "PINN", "inside", ""),
         ("B", "Hybrid-calibrated", "Calibrated", "outside", ""),
         ("B", "GP+law", "Calibrated", "outside", "")]
PAIRED = []
for ocean, a, b, region, title in PAIRS:
    if title:
        print("\n" + title)
    for n in N_TRAIN:
        d, k, p = paired(ocean, a, b, n, "rmse_" + region)
        PAIRED.append(dict(ocean=ocean, a=a, b=b, region=region, n=n, diff=d, wins=k, p=p))
        print(f"   ocean {ocean}, {region:7s}, N = {n:3d}: {LABELS[a]} minus {LABELS[b]} = {d:+6.2f} dB, "
              f"first better in {k}/{len(SEEDS)} seeds (sign test p = {p:.3f})")

# %% [markdown]
# ## 9. Why the GP length scale is bounded
#
# The same GP + law is refitted on ocean A with the range length scale left free (up to 50 km), for every $N$ and seed. The free GP extrapolates worse with few samples, because its long-range component competes with the law. The results of the whole run are then written to `results/`.

# %%
FREE_BOUNDS = list(GP_BOUNDS); FREE_BOUNDS[1] = (np.log(0.02), np.log(5.0))
ABL = []
for n in N_TRAIN:
    for s in SEEDS:
        l, f, y = sample_train(n, np.random.default_rng(s), true_tl_A)
        g = KrigingGP(l, f, y, "law", s, bounds=FREE_BOUNDS)
        ABL.append(dict(n=n, seed=s, rmse_inside=rms(g.predict(*TESTS["inside"]) - true_tl_A(*TESTS["inside"])),
                        rmse_outside=rms(g.predict(*TESTS["outside"]) - true_tl_A(*TESTS["outside"])),
                        k=float(g.k), ls_l_km=float(g.ls[0] * L_MAX)))
print("Ocean A, GP + law, mean over seeds: RMSE inside / outside (dB), fitted n_s, range length scale (km)")
for n in N_TRAIN:
    fr = [a for a in ABL if a["n"] == n]
    print(f"   N = {n:3d}: free {np.mean([a['rmse_inside'] for a in fr]):.2f} / {np.mean([a['rmse_outside'] for a in fr]):.2f}, "
          f"n_s {np.mean([a['k'] for a in fr]):.2f}, l {np.mean([a['ls_l_km'] for a in fr]):5.1f} km   |   bounded "
          f"{stat('A', 'GP+law', n, 'rmse_inside')[0]:.2f} / {stat('A', 'GP+law', n, 'rmse_outside')[0]:.2f}, "
          f"n_s {stat('A', 'GP+law', n, 'k')[0]:.2f}, l {stat('A', 'GP+law', n, 'gp_ls_l_km')[0]:4.1f} km")

json.dump(dict(mode=MODE, seeds=SEEDS, n_train=N_TRAIN, runtime_s=RUNTIME, sanity=SANITY,
               textbook_rmse=TEXTBOOK_RMSE, oracle_rmse_A=ORACLE_RMSE_A, runs=RUNS, paired=PAIRED, gp_free_ablation=ABL,
               settings=dict(gp_ls_range_max_km=GP_LS_RANGE_MAX * L_MAX, ocean_B=dict(z_src_m=Z_SRC_M, z_rcv_m=Z_RCV_M,
                             wave_rms_m=WAVE_RMS_M, c_sound=C_SOUND)),
               versions=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__)),
          open(OUT / f"example1_results_{MODE}.json", "w"), indent=1)
print("Written:", OUT / f"example1_results_{MODE}.json")

# %% [markdown]
# ## 10. Check against the article (Section II-F)
#
# Every number that Section II-F states is recomputed here from the runs above. A row is flagged when the run differs from the article by more than 5% and by more than the rounding of the printed value. In `quick` mode, only the rows that do not depend on the seeds are checked.

# %%
def row(label, paper, value, dec=1, seed_dependent=True):
    return dict(label=label, paper=paper, value=value, dec=dec, dep=seed_dependent)


mean = lambda o, kind, n, key: stat(o, kind, n, key)[0]
io, oo = "rmse_inside", "rmse_outside"


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 else float("nan")


free = lambda n, key: float(np.mean([a[key] for a in ABL if a["n"] == n]))
ROWS = [row("Pipeline check: PINN n_s (clean data)", 1.71, SANITY["pinn_k"], 2, False),
        row("Pipeline check: PINN zeta (clean data)", 1.25, SANITY["pinn_gamma"], 2, False),
        row("A: textbook law alone, RMSE inside", 11.5, TEXTBOOK_RMSE["A"]["inside"], 1, False),
        row("A: textbook law alone, RMSE outside", 20.3, TEXTBOOK_RMSE["A"]["outside"], 1, False),
        row("A outside: MLP, lowest over N", 23.4, min(mean("A", "MLP", n, oo) for n in N_TRAIN)),
        row("A outside: MLP, highest over N", 29.8, max(mean("A", "MLP", n, oo) for n in N_TRAIN)),
        row("A outside: textbook hybrid, lowest for N>=30", 6.2, min(mean("A", "Hybrid-textbook", n, oo) for n in (30, 100, 300))),
        row("A outside: textbook hybrid, highest for N>=30", 6.8, max(mean("A", "Hybrid-textbook", n, oo) for n in (30, 100, 300)))]
for kind, vals in (("Hybrid-calibrated", (5.0, 2.7, 1.3)), ("Hybrid-joint", (4.8, 2.1, 1.1)), ("PINN", (5.4, 2.4, 1.0))):
    ROWS += [row(f"A outside, N={n}: {kind}", p, mean("A", kind, n, oo)) for n, p in zip((30, 100, 300), vals)]
ROWS += [row(f"A outside, N={n}: calibrated law", p, mean("A", "Calibrated", n, oo)) for n, p in zip(N_TRAIN, (4.7, 1.2, 1.0, 1.2))]
ROWS += [row("A, N=30: calibrated hybrid inside", 2.8, mean("A", "Hybrid-calibrated", 30, io)),
         row("A, N=30: calibrated hybrid outside", 5.0, mean("A", "Hybrid-calibrated", 30, oo)),
         row("A, N=30: calibrated law inside", 1.0, mean("A", "Calibrated", 30, io)),
         row("A, N=30: calibrated law outside", 1.2, mean("A", "Calibrated", 30, oo)),
         row("A inside, N=100: GP + law", 0.52, mean("A", "GP+law", 100, io), 2),
         row("A inside, N=300: GP + law", 0.33, mean("A", "GP+law", 300, io), 2),
         row("A inside, N=100: calibrated law", 0.88, mean("A", "Calibrated", 100, io), 2),
         row("A inside, N=300: calibrated law", 0.86, mean("A", "Calibrated", 300, io), 2),
         row("A inside: exact law (floor)", 0.86, ORACLE_RMSE_A["inside"], 2, False),
         row("A outside, N=100: GP + law", 0.84, mean("A", "GP+law", 100, oo), 2),
         row("A outside, N=300: GP + law", 1.17, mean("A", "GP+law", 300, oo), 2),
         row("A: GP + law coverage inside, lowest for N>=30 (%)", 88, 100 * min(mean("A", "GP+law", n, "cover_inside") for n in (30, 100, 300)), 0),
         row("A: GP + law coverage inside, highest for N>=30 (%)", 96, 100 * max(mean("A", "GP+law", n, "cover_inside") for n in (30, 100, 300)), 0),
         row("A: constant-mean GP coverage, both shifts, lowest (%)", 4, 100 * min(mean("A", "GP-const", n, "cover_both") for n in N_TRAIN), 0),
         row("A: constant-mean GP coverage, both shifts, highest (%)", 25, 100 * max(mean("A", "GP-const", n, "cover_both") for n in N_TRAIN), 0),
         row("A outside, N=30: GP + law with a free length scale", 3.6, free(30, "rmse_outside")),
         row("A outside, N=30: GP + law with the bound", 1.8, mean("A", "GP+law", 30, oo)),
         row("A, N=30: Spearman(n_s, zeta) of the PINN over seeds", -0.90,
             spearman([r["k"] for r in runs("A", "PINN", 30)], [r["gamma"] for r in runs("A", "PINN", 30)]), 2),
         row("B inside: calibrated law, lowest over N", 3.0, min(mean("B", "Calibrated", n, io) for n in N_TRAIN)),
         row("B inside: calibrated law, highest over N", 3.4, max(mean("B", "Calibrated", n, io) for n in N_TRAIN)),
         row("B inside: PINN, lowest for N>=100", 2.3, min(mean("B", "PINN", n, io) for n in (100, 300))),
         row("B inside: PINN, highest for N>=100", 2.3, max(mean("B", "PINN", n, io) for n in (100, 300))),
         row("B inside, N=300: calibrated hybrid", 0.85, mean("B", "Hybrid-calibrated", 300, io), 2),
         row("B inside, N=300: GP + law", 1.26, mean("B", "GP+law", 300, io), 2)]
ROWS += [row(f"B outside, N=300: {k}", p, mean("B", k, 300, oo))
         for k, p in (("Hybrid-calibrated", 2.9), ("Calibrated", 3.9), ("GP+law", 4.7), ("PINN", 8.1), ("MLP", 23.8))]
ROWS += [row("B, N=300: GP + law coverage outside, lowest region (%)", 91,
             100 * min(mean("B", "GP+law", 300, "cover_" + k) for k in ("farther", "higher_f", "both")), 0),
         row("B, N=300: GP + law coverage outside, highest region (%)", 97,
             100 * max(mean("B", "GP+law", 300, "cover_" + k) for k in ("farther", "higher_f", "both")), 0)]
n_flag = n_chk = 0
print(f"{'Quantity (Section II-F)':54s} {'paper':>7s} {'this run':>9s} {'diff':>7s}  status")
for r in ROWS:
    rel = abs(r["value"] - r["paper"]) / abs(r["paper"])
    bad = rel > 0.05 and abs(r["value"] - r["paper"]) > 0.5 * 10 ** (-r["dec"])   # beyond 5% and beyond the rounding
    checked = (not r["dep"]) or MODE == "full"
    status = "info (quick mode)" if not checked else ("FLAG > 5%" if bad else "ok")
    n_flag += checked and bad; n_chk += checked
    print(f"{r['label']:54s} {r['paper']:7.{r['dec']}f} {r['value']:9.{r['dec'] + 1}f} {100 * rel:6.1f}%  {status}")
print(f"\n{n_chk} rows checked, {n_flag} flagged.")

# %% [markdown]
# ## 11. Figure of the article
#
# `figures/fig_example1.pdf` is Fig. 4 of the article (IEEE single-column width, four panels).
#
# - **(a)** Ocean A: RMSE outside the measured region versus $N$ (mean over seeds, bars from the best to the worst seed). Dotted: textbook law alone. Dash-dotted: exact law, the floor of any model without the ripple.
# - **(b)** Ocean A at 35 kHz, outside the measured band, $N = 100$, seed 0: prediction error versus range, with the 95% band of the GP + law.
# - **(c)** Ocean B: RMSE inside and outside with $N = 100$, mean over seeds.
# - **(d)** Ocean B at 5 kHz, $N = 100$, seed 0: prediction error versus range.

# %%
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d6", "lines.linewidth": 1.2, "savefig.bbox": "tight",
    "pdf.fonttype": 42})                          # TrueType fonts in the PDF
COL_W = 3.5
INK, MUTED, BAND = "#0b0b0b", "#6b6a66", "#c9d6df"
STYLE = {"MLP": dict(color="#2a78d6", marker="o", ls="-"),
         "PINN": dict(color="#eb6834", marker="s", ls="--"),
         "Hybrid-textbook": dict(color="#1baf7a", marker="^", ls="-."),
         "Hybrid-calibrated": dict(color="#9b59b6", marker="v", ls=(0, (1, 1))),
         "Calibrated": dict(color="#eda100", marker="D", ls=(0, (5, 1.5, 1, 1.5))),
         "GP+law": dict(color="#264653", marker="P", ls="-")}
FIG_KINDS = list(STYLE)
SHORT = {"MLP": "MLP (data only)", "PINN": "PINN", "Hybrid-textbook": "Hybrid, textbook law",
         "Hybrid-calibrated": "Hybrid, calibrated law", "Calibrated": "Calibrated law (LS)", "GP+law": "GP + law"}


def style_axes(ax):
    ax.grid(True, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, width=0.6, length=2.5)


fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.15))
# (a) ocean A, RMSE outside versus N
ax = axes[0, 0]
shifts = np.exp(np.linspace(-0.15, 0.15, len(FIG_KINDS)))
for kind, sh in zip(FIG_KINDS, shifts):
    per = [[r["rmse_outside"] for r in runs("A", kind, n)] for n in N_TRAIN]
    m = np.array([np.mean(v) for v in per])
    ax.errorbar(np.array(N_TRAIN) * sh, m, yerr=[m - [min(v) for v in per], [max(v) for v in per] - m],
                color=STYLE[kind]["color"], marker=STYLE[kind]["marker"], ls=STYLE[kind]["ls"], ms=3, capsize=1.2,
                elinewidth=0.5, lw=1.0, label=SHORT[kind])
ax.axhline(TEXTBOOK_RMSE["A"]["outside"], color=MUTED, ls=":", lw=1.0, label="Textbook law alone")
ax.axhline(ORACLE_RMSE_A["outside"], color=MUTED, ls="-.", lw=0.8, label="Exact law (floor)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(N_TRAIN); ax.set_xticklabels([str(n) for n in N_TRAIN])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("Measurements $N$"); ax.set_ylabel("RMSE outside (dB)"); ax.set_title("(a) Ocean A: right form", pad=3)
style_axes(ax)


def error_panel(ax, models, ocean, fq, title):
    # prediction error (model minus truth) versus range at one frequency; GP + law with its 95% band
    lg = np.linspace(L_MIN, L_MAX, 300); fg = np.full_like(lg, fq); truth = OCEANS[ocean](lg, fg)
    if fq <= F_TRAIN:
        ax.axvspan(L_MIN, L_TRAIN, color="#efefec", lw=0)
    mu, sd = models[(ocean, "GP+law")].predict(lg, fg, return_std=True)
    ax.fill_between(lg, mu - truth - 1.96 * sd, mu - truth + 1.96 * sd, color=BAND, lw=0, label="GP + law, 95% band")
    ax.axhline(0, color=INK, lw=1.2)                                   # truth = zero error
    for kind in FIG_KINDS:
        ax.plot(lg, predict(models[(ocean, kind)], lg, fg) - truth, color=STYLE[kind]["color"], ls=STYLE[kind]["ls"], lw=0.9)
    ax.set_ylim(-12, 12); ax.set_xlim(0, L_MAX)
    ax.set_xlabel("Range (km)"); ax.set_ylabel("Error (dB)"); ax.set_title(title, pad=3); style_axes(ax)


# (b) ocean A at 35 kHz, N = 100
error_panel(axes[0, 1], KEEP, "A", 35.0, "(b) Ocean A, 35 kHz, $N$ = 100")
# (c) ocean B, RMSE inside and outside, N = 100
ax = axes[1, 0]
width = 0.8 / len(FIG_KINDS)
for i, kind in enumerate(FIG_KINDS):
    x = np.arange(2) - 0.4 + (i + 0.5) * width
    vals = [stat("B", kind, 100, "rmse_" + k) for k in ("inside", "outside")]
    m = np.array([v[0] for v in vals])
    ax.bar(x, m, width, color=STYLE[kind]["color"], lw=0)
    ax.errorbar(x, m, yerr=[m - [v[2] for v in vals], [v[3] for v in vals] - m], fmt="none", ecolor=INK,
                elinewidth=0.5, capsize=1.0)
ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks([0, 1]); ax.set_xticklabels(["Inside", "Outside"]); ax.set_ylabel("RMSE (dB)")
ax.set_title("(c) Ocean B: wrong form, $N$ = 100", pad=3); style_axes(ax); ax.grid(False, axis="x")
# (d) ocean B at 5 kHz, N = 100
error_panel(axes[1, 1], KEEP, "B", 5.0, "(d) Ocean B, 5 kHz, $N$ = 100")

handles, labels = axes[0, 0].get_legend_handles_labels()
hb, lb = axes[0, 1].get_legend_handles_labels()
fig.legend(handles + hb, labels + lb, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12),
           handlelength=2.2, columnspacing=0.8, fontsize=6.2)
fig.tight_layout(h_pad=0.4, w_pad=0.6)
fig.savefig(FIG / "fig_example1.pdf"); plt.show()
print("Written:", FIG / "fig_example1.pdf")

# %% [markdown]
# ## 12. Reading the results
#
# The numbers below are those of the full run on Colab (5 seeds), which are the numbers of the article.
#
# **Ocean A (right form).**
#
# 1. With the textbook law, the hybrid stays between 6.2 and 6.8 dB outside the measured region for $N \ge 30$. With its law calibrated first, it reaches 5.0, 2.7, and 1.3 dB for $N$ = 30, 100, and 300, and the PINN reaches 5.4, 2.4, and 1.0 dB. With calibrated physics on both sides, mechanisms R and L are not distinguishable.
# 2. The calibrated law alone reaches 1.2, 1.0, and 1.2 dB outside for $N$ = 30, 100, and 300, and no learner improves on it outside in more than 4 of 5 seeds. The gain comes from the law and its calibration.
# 3. A network correction without a prior overfits: at $N = 30$, the calibrated hybrid has 2.8 dB inside and 5.0 dB outside, against 1.0 and 1.2 dB for the calibrated law alone.
# 4. From $N = 100$, the GP + law learns the ripple inside (0.52 and 0.33 dB, against 0.88 and 0.86 dB for the calibrated law) and equals the law outside. With $N \le 30$ it brings no gain.
# 5. The 95% intervals of the GP + law cover 88 to 96% of the inside points for $N \ge 30$. Those of the data-only GP contain the truth for only 4 to 25% of the points under both shifts.
#
# **Ocean B (wrong form).**
#
# 6. The calibrated law stays between 3.0 and 3.4 dB inside whatever $N$, and the PINN at 2.3 dB for $N \ge 100$, since its residual enforces the wrong law. The calibrated hybrid reaches 0.85 dB and the GP + law 1.26 dB at $N = 300$.
# 7. Outside the data, no model is accurate. At $N = 300$: calibrated hybrid 2.9 dB, calibrated law 3.9 dB, GP + law 4.7 dB, PINN 8.1 dB, MLP 23.8 dB. The PINN extrapolates worse than the law that it encodes, and the band of the GP + law still covers 91 to 97% of the outside points.
#
# **Limits.** Both oceans are simulated. The law has a closed form with three linear parameters, which favors calibration by least squares, and a PDE without a closed-form solution is not tested. Ocean B adds one physical effect with a fixed sea state. The networks are trained without weight decay or early stopping. With five seeds, differences are reported as consistent or not across seeds.

# %% [markdown]
# ## 13. Try it yourself
#
# Change one parameter and rerun the cell (about 30 s, $N = 30$, seed 0).
#
# - `WAVE_RMS_TRY`: rms wave height of ocean B. A calm sea (0.05 m) deepens the interference nulls, and a rough sea (1 m) smooths them.
# - `WEIGHT_DECAY_TRY`: decoupled weight decay of the calibrated hybrid (0 in the article). Values around 1 shrink the correction toward zero away from the data.

# %%
WAVE_RMS_TRY = 0.05          # rms wave height in m (article: 0.5)
WEIGHT_DECAY_TRY = 1.0       # weight decay (article: 0)


def fit_hybrid_wd(l, f, y, seed, wd):
    torch.manual_seed(seed); model = Hybrid("calibrated", Calibrated(l, f, y)); lt, ft, yt = to_t(l, f, y)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for _ in range(STEPS):
        opt.zero_grad(); loss = torch.mean(((model(lt, ft) - yt) / 10.0) ** 2); loss.backward(); opt.step(); sched.step()
    return model


_saved = WAVE_RMS_M
try:
    WAVE_RMS_M = WAVE_RMS_TRY
    l, f, y = sample_train(30, np.random.default_rng(0), true_tl_B)
    print(f"Ocean B with rms wave height {WAVE_RMS_TRY:g} m, N = 30, seed 0. RMSE inside / outside (dB):")
    for kind in ("Calibrated", "PINN", "Hybrid-calibrated", "GP+law"):
        m = FITTERS[kind](l, f, y, 0)
        print(f"   {LABELS[kind]:34s} {rms(predict(m, *TESTS['inside']) - true_tl_B(*TESTS['inside'])):5.2f} / "
              f"{rms(predict(m, *TESTS['outside']) - true_tl_B(*TESTS['outside'])):5.2f}")
finally:
    WAVE_RMS_M = _saved
l, f, y = sample_train(30, np.random.default_rng(0), true_tl_A)
for wd in (0.0, WEIGHT_DECAY_TRY):
    m = fit_hybrid_wd(l, f, y, 0, wd)
    print(f"Ocean A, calibrated hybrid with weight decay {wd:g}: RMSE inside "
          f"{rms(predict(m, *TESTS['inside']) - true_tl_A(*TESTS['inside'])):.2f} / outside "
          f"{rms(predict(m, *TESTS['outside']) - true_tl_A(*TESTS['outside'])):.2f} dB")
