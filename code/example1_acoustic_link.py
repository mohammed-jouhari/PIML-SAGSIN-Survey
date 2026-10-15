"""Example 1: learning the transmission loss of an underwater acoustic link from sparse data.

Three models are compared:
  (a) MLP     : a plain neural network trained on data only;
  (b) PINN    : the same network, trained with the data loss plus the residuals of the distance and
                frequency derivatives of the Stojanovic model, with the spreading factor k and the
                absorption scale gamma learned as physical parameters;
  (c) Hybrid  : nominal physics model (k = 1.5, gamma = 1) plus a neural network that learns the
                correction from data.

The "true" channel follows the Stojanovic form with parameters that differ from the nominal ones
(k = 1.7, gamma = 1.25, offset 3 dB) plus a bottom-interaction ripple that no physics model here
contains. Training data are noisy samples (sigma = 1.5 dB) inside l <= 5 km and f <= 25 kHz.
Errors are measured against the noise-free truth inside that region (interpolation) and outside it
(extrapolation in distance or frequency).

Run:  python example1_acoustic_link.py      (about 11 minutes on a 2-thread CPU)
"""
import json, time
from pathlib import Path

import numpy as np
import torch

from physics import thorp_db_per_km, transmission_loss_db

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- problem definition
L_MIN, L_MAX, F_MIN, F_MAX = 0.2, 10.0, 2.0, 40.0          # full domain (km, kHz)
L_TRAIN, F_TRAIN = 5.0, 25.0                               # training region: l <= 5 km, f <= 25 kHz
TRUE = dict(k=1.7, gamma=1.25, c0=3.0)                     # "true" environment
NOMINAL = dict(k=1.5, gamma=1.0, c0=0.0)                   # textbook prior
RIPPLE_DB, RIPPLE_PERIOD_KM, RIPPLE_DECAY_KM = 2.0, 1.5, 4.0
NOISE_DB = 1.5
N_TRAIN = [10, 30, 100, 300]
SEEDS = [0, 1, 2, 3, 4]
STEPS, LR, N_COLLOC, LAMBDA = 3000, 2e-3, 1000, 1.0       # Adam steps for MLP and Hybrid
PINN_ADAM_STEPS, PINN_LBFGS_ITER = 2000, 400
TL_SCALE = 100.0                                           # outputs are learned in units of 100 dB


def true_tl(l, f):
    ripple = RIPPLE_DB * np.sin(2 * np.pi * l / RIPPLE_PERIOD_KM) * np.exp(-l / RIPPLE_DECAY_KM)
    return transmission_loss_db(l, f, **TRUE) + ripple


def sample_train(n, rng):
    l = rng.uniform(L_MIN, L_TRAIN, n)
    f = rng.uniform(F_MIN, F_TRAIN, n)
    return l, f, true_tl(l, f) + rng.normal(0, NOISE_DB, n)


def test_sets(rng, n=4000):
    l = rng.uniform(L_MIN, L_MAX, n)
    f = rng.uniform(F_MIN, F_MAX, n)
    inside = (l <= L_TRAIN) & (f <= F_TRAIN)
    return {"interpolation": (l[inside], f[inside]), "extrapolation": (l[~inside], f[~inside])}


# ---------------------------------------------------------------- models
def thorp_t(f_khz):
    f2 = f_khz ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def features(l, f):
    # Distance enters on a log scale (spreading loss is logarithmic in l); same features for all models.
    return torch.stack([(torch.log(l) - np.log(L_MIN)) / (np.log(L_MAX) - np.log(L_MIN)), f / F_MAX], dim=-1)


class Net(torch.nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(),
                                        torch.nn.Linear(width, width), torch.nn.Tanh(),
                                        torch.nn.Linear(width, 1))

    def forward(self, l, f):
        return self.body(features(l, f)).squeeze(-1) * TL_SCALE


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = Net()

    def forward(self, l, f):
        return self.net(l, f)


class PINN(torch.nn.Module):
    """Network output TL(l, f) with two physics residuals taken from the Stojanovic model:
         r_x = l dTL/dl - [10 k / ln 10 + gamma * l * alpha(f)]     (derivative w.r.t. ln l)
         r_f = dTL/df   - gamma * l * dalpha/df
    Together they fix TL up to one constant, which the data set. k and gamma are learned.
    Writing the distance residual in ln l turns the spreading term into a constant and keeps the
    residual well scaled over the 0.2-10 km range."""
    def __init__(self):
        super().__init__(); self.net = Net()
        self.k = torch.nn.Parameter(torch.tensor(NOMINAL["k"]))
        self.gamma = torch.nn.Parameter(torch.tensor(NOMINAL["gamma"]))

    def forward(self, l, f):
        return self.net(l, f)

    def residual(self, l, f):
        l = l.clone().requires_grad_(True)
        f = f.clone().requires_grad_(True)
        tl = self.net(l, f)
        dtl_dl, dtl_df = torch.autograd.grad(tl.sum(), (l, f), create_graph=True)
        alpha = thorp_t(f)
        dalpha_df = torch.autograd.grad(alpha.sum(), f, create_graph=True)[0]
        r_x = l * dtl_dl - (10 * self.k / np.log(10) + self.gamma * l * alpha)
        r_f = dtl_df - self.gamma * l * dalpha_df
        return torch.cat([r_x, r_f])


class Hybrid(torch.nn.Module):
    """Nominal physics plus a learned correction."""
    def __init__(self):
        super().__init__(); self.net = Net()

    def forward(self, l, f):
        phys = 10 * NOMINAL["k"] * torch.log10(l / 1e-3) + NOMINAL["gamma"] * l * thorp_t(f) + NOMINAL["c0"]
        return phys + self.net(l, f) / 10.0      # correction head works at a finer scale (units of 10 dB)


def train(kind, l, f, y, seed):
    """MLP and Hybrid: Adam with cosine decay (L-BFGS made these data-only models unstable on sparse data).
    PINN: Adam, then L-BFGS, the usual PINN schedule; without the L-BFGS stage the residual is not
    resolved well enough to identify k and gamma (see sanity_check)."""
    torch.manual_seed(seed)
    model = {"MLP": MLP, "PINN": PINN, "Hybrid": Hybrid}[kind]()
    lt, ft, yt = (torch.tensor(a, dtype=torch.float32) for a in (l, f, y))
    if kind == "PINN":
        g = torch.Generator().manual_seed(1000 + seed)
        lc = L_MIN + (L_MAX - L_MIN) * torch.rand(N_COLLOC, generator=g)   # collocation over the FULL domain
        fc = F_MIN + (F_MAX - F_MIN) * torch.rand(N_COLLOC, generator=g)

    def loss_fn():
        loss = torch.mean(((model(lt, ft) - yt) / 10.0) ** 2)            # data term, dB scaled by 10
        if kind == "PINN":
            loss = loss + LAMBDA * torch.mean((model.residual(lc, fc) / 10.0) ** 2)
        return loss

    steps = PINN_ADAM_STEPS if kind == "PINN" else STEPS
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step(); sched.step()
    if kind == "PINN":
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=PINN_LBFGS_ITER, history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            lb.zero_grad(); loss = loss_fn(); loss.backward(); return loss
        lb.step(closure)
    return model


def sanity_check():
    """Dense, noise-free data over the whole domain and no ripple: the PINN should recover k and gamma."""
    global RIPPLE_DB
    saved, RIPPLE_DB = RIPPLE_DB, 0.0
    rng = np.random.default_rng(0)
    l, f = rng.uniform(L_MIN, L_MAX, 300), rng.uniform(F_MIN, F_MAX, 300)
    m = train("PINN", l, f, true_tl(l, f), 0)
    RIPPLE_DB = saved
    return dict(k_learned=float(m.k.detach()), gamma_learned=float(m.gamma.detach()), k_true=TRUE["k"], gamma_true=TRUE["gamma"])


def rmse(model, l, f):
    with torch.no_grad():
        p = model(torch.tensor(l, dtype=torch.float32), torch.tensor(f, dtype=torch.float32)).numpy()
    return float(np.sqrt(np.mean((p - true_tl(l, f)) ** 2)))


def predict(model, l, f):
    with torch.no_grad():
        return model(torch.tensor(l, dtype=torch.float32), torch.tensor(f, dtype=torch.float32)).numpy()


# ---------------------------------------------------------------- experiment
def main():
    t0 = time.time()
    tests = test_sets(np.random.default_rng(123))
    nominal_rmse = {k: float(np.sqrt(np.mean((transmission_loss_db(l, f, **NOMINAL) - true_tl(l, f)) ** 2)))
                    for k, (l, f) in tests.items()}
    res = {"config": dict(true=TRUE, nominal=NOMINAL, ripple_db=RIPPLE_DB, noise_db=NOISE_DB, n_train=N_TRAIN,
                          seeds=SEEDS, steps=STEPS, lr=LR, n_colloc=N_COLLOC, lam=LAMBDA,
                          pinn_adam_steps=PINN_ADAM_STEPS, pinn_lbfgs_iter=PINN_LBFGS_ITER,
                          train_region=dict(l_max_km=L_TRAIN, f_max_khz=F_TRAIN),
                          domain=dict(l_km=[L_MIN, L_MAX], f_khz=[F_MIN, F_MAX])),
           "nominal_physics_only_rmse_db": nominal_rmse, "runs": [], "curves": {}}
    res["sanity_check_dense_noise_free"] = sanity_check()
    print("sanity check:", res["sanity_check_dense_noise_free"], flush=True)
    keep = {}
    for n in N_TRAIN:
        for s in SEEDS:
            l, f, y = sample_train(n, np.random.default_rng(s))
            for kind in ("MLP", "PINN", "Hybrid"):
                m = train(kind, l, f, y, s)
                run = dict(model=kind, n=n, seed=s, **{f"rmse_{k}": rmse(m, *v) for k, v in tests.items()})
                if kind == "PINN":
                    run.update(k_learned=float(m.k.detach()), gamma_learned=float(m.gamma.detach()))
                res["runs"].append(run)
                if n == 30 and s == 0:
                    keep[kind] = m
                    if kind == "MLP":
                        res["curves"]["train_points_n30_seed0"] = dict(l=l.tolist(), f=f.tolist(), y=y.tolist())
            print(f"n={n:4d} seed={s} done  ({time.time() - t0:5.0f} s)", flush=True)
    # curves for the illustration figure
    lg = np.linspace(L_MIN, L_MAX, 300)
    for fq in (20.0, 35.0):
        fg = np.full_like(lg, fq)
        res["curves"][f"f{int(fq)}"] = dict(l=lg.tolist(), truth=true_tl(lg, fg).tolist(),
                                            nominal=transmission_loss_db(lg, fg, **NOMINAL).tolist(),
                                            **{k: predict(m, lg, fg).tolist() for k, m in keep.items()})
    # summary
    summ = {}
    for kind in ("MLP", "PINN", "Hybrid"):
        for n in N_TRAIN:
            rs = [r for r in res["runs"] if r["model"] == kind and r["n"] == n]
            d = {k: [float(np.mean([r[f"rmse_{k}"] for r in rs])), float(np.std([r[f"rmse_{k}"] for r in rs]))]
                 for k in tests}
            if kind == "PINN":
                d["k"] = [float(np.mean([r["k_learned"] for r in rs])), float(np.std([r["k_learned"] for r in rs]))]
                d["gamma"] = [float(np.mean([r["gamma_learned"] for r in rs])), float(np.std([r["gamma_learned"] for r in rs]))]
            summ[f"{kind}_n{n}"] = d
    res["summary_mean_std"] = summ
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "example1_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"nominal_physics_only_rmse_db": nominal_rmse, "summary": summ, "runtime_s": res["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
