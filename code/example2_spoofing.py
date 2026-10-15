"""Example 2: physics-consistency detection of spoofed delay reports in an underwater acoustic network.

A node reports two-way travel times (halved) to K = 6 anchors with known positions: four surface buoys
and two seabed nodes. An attacker manipulates the reports. Two detectors are compared:

  (a) Physics residual detector. It fits the node position p and the effective slowness s = 1/c by
      Gauss-Newton on tau_i = s * ||p - a_i||. Six reports and four unknowns leave two degrees of
      freedom, so a consistent report set fits well and an inconsistent one does not. The score is
      the normalized residual sum of squares. No training data is needed, and the sound speed is
      estimated jointly, so a change of water temperature is absorbed by the fit.
  (b) Data-driven detector. An MLP classifier that receives the six delays and is trained, for one
      deployment (fixed anchor layout E1), on normal reports and single-link delay injections.

Deployments: E1 shallow (2 km x 2 km, 200 m deep, c in [1490, 1510] m/s), used for training;
             E2 deep and wide (4 km x 4 km, 1000 m deep, c in [1515, 1535] m/s), a new deployment.
Each deployment has one fixed anchor layout; node position and sound speed vary from report to report.
Noise: Gaussian timing noise (0.5 ms) plus occasional multipath excess delay (prob. 0.05, up to 2 ms).
Attacks:  A1 single-link delay injection, +3 to +15 ms (seen in training)
          A2 two links, +-1.5 to +-5 ms each (not seen in training)
          A3 position spoofing: all delays computed from a fake position 100-400 m away (physics-consistent)

Run:  python example2_spoofing.py      (about 7 minutes on a 2-thread CPU)
"""
import json, time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

K = 6
SIGMA = 0.5e-3                      # timing noise std (s)
P_MULTI, MULTI_MAX = 0.05, 2.0e-3   # multipath excess delay
ENVS = {"E1": dict(L=2000.0, D=200.0, c=(1490.0, 1510.0), layout_seed=11),
        "E2": dict(L=4000.0, D=1000.0, c=(1515.0, 1535.0), layout_seed=22)}
N_TRAIN, N_TEST = 50000, 4000
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, BATCH, LR, WIDTH = 150, 256, 3e-3, 256   # tuned so the MLP learns the seen case well


# ---------------------------------------------------------------- scenario generation
def layout(env):
    """One fixed anchor layout per deployment: four buoys near the corners, two seabed nodes."""
    r = np.random.default_rng(env["layout_seed"])
    L, D = env["L"], env["D"]
    j = 0.075 * L
    corners = np.array([[0, 0], [L, 0], [0, L], [L, L]], dtype=float)
    buoys = np.concatenate([corners + r.uniform(-j, j, (4, 2)), np.full((4, 1), -5.0)], axis=1)
    bed = np.concatenate([r.uniform(0.2 * L, 0.8 * L, (2, 2)), np.full((2, 1), -D)], axis=1)
    return np.concatenate([buoys, bed], axis=0)                          # (K, 3)


def geometry(env, n, rng):
    L, D = env["L"], env["D"]
    anchors = np.repeat(layout(env)[None], n, axis=0)                    # (n, K, 3), fixed deployment
    node = np.stack([rng.uniform(0.1 * L, 0.9 * L, n), rng.uniform(0.1 * L, 0.9 * L, n),
                     rng.uniform(-0.9 * D, -0.1 * D, n)], axis=1)        # (n, 3)
    c = rng.uniform(*env["c"], n)
    return anchors, node, c


def delays(anchors, pos, c, rng):
    d = np.linalg.norm(anchors - pos[:, None, :], axis=2)
    tau = d / c[:, None] + rng.normal(0, SIGMA, d.shape)
    multi = rng.random(d.shape) < P_MULTI
    return tau + multi * rng.uniform(0, MULTI_MAX, d.shape)


def make_set(env_name, attack, n, rng):
    """Return delays, anchors, labels. Half normal, half attacked (attack None -> all normal)."""
    env = ENVS[env_name]
    anchors, node, c = geometry(env, n, rng)
    if attack == "A3":
        ang = rng.uniform(0, 2 * np.pi, n); r = rng.uniform(100, 400, n)
        fake = node.copy(); fake[:, 0] += r * np.cos(ang); fake[:, 1] += r * np.sin(ang)
        y = (np.arange(n) >= n // 2).astype(int)
        pos = np.where(y[:, None] == 1, fake, node)
        return delays(anchors, pos, c, rng), anchors, y
    tau = delays(anchors, node, c, rng)
    y = np.zeros(n, dtype=int)
    if attack is None:
        return tau, anchors, y
    y[n // 2:] = 1
    idx = np.where(y == 1)[0]
    if attack == "A1":
        link = rng.integers(0, K, idx.size)
        tau[idx, link] += rng.uniform(3e-3, 15e-3, idx.size)
    elif attack == "A2":
        for i in idx:
            links = rng.choice(K, 2, replace=False)
            tau[i, links] += rng.choice([-1, 1], 2) * rng.uniform(1.5e-3, 5e-3, 2)
    return tau, anchors, y


# ---------------------------------------------------------------- detector (a): physics residual
def physics_fit(tau, anchors, iters=25):
    """Damped Gauss-Newton for theta = (x, y, z, s), with tau_i = s * ||p - a_i||. Vectorized over samples."""
    n = tau.shape[0]
    p = anchors.mean(axis=1).copy()
    p[:, 2] = anchors[:, :, 2].mean(axis=1) * 0.5
    s = np.full(n, 1 / 1500.0)
    lam = np.full(n, 1e-3)
    def resid(p, s):
        d = np.linalg.norm(anchors - p[:, None, :], axis=2)
        return tau - s[:, None] * d, d
    r, d = resid(p, s)
    cost = (r ** 2).sum(1)
    for _ in range(iters):
        u = (p[:, None, :] - anchors) / d[..., None]                      # unit vectors (n, K, 3)
        J = np.concatenate([s[:, None, None] * u, d[..., None]], axis=2)  # d tau / d theta  (n, K, 4)
        scale = np.array([1.0, 1.0, 1.0, 1e3])                             # condition the slowness column
        Js = J * (1 / scale)
        A = np.einsum("nki,nkj->nij", Js, Js) + lam[:, None, None] * np.eye(4)
        g = np.einsum("nki,nk->ni", Js, r)
        step = np.linalg.solve(A, g[..., None])[..., 0] / scale
        p_new, s_new = p + step[:, :3], s + step[:, 3]
        r_new, d_new = resid(p_new, s_new)
        c_new = (r_new ** 2).sum(1)
        ok = c_new < cost
        p[ok], s[ok], r[ok], d[ok], cost[ok] = p_new[ok], s_new[ok], r_new[ok], d_new[ok], c_new[ok]
        lam = np.where(ok, lam * 0.3, lam * 10)
    return p, s, cost / SIGMA ** 2


# ---------------------------------------------------------------- detector (b): data-driven MLP
def features(tau, anchors):
    # The anchors are fixed within a deployment, so the learned detector sees the six delays only.
    return tau.astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, d_in, w=WIDTH):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d_in, w), torch.nn.SiLU(), torch.nn.Linear(w, w), torch.nn.SiLU(),
                                     torch.nn.Linear(w, w), torch.nn.SiLU(), torch.nn.Linear(w, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train_mlp(X, y, seed):
    """Adam with a one-cycle learning-rate schedule. A shorter or smaller set-up (40 epochs, width 128)
    did not learn the seen case (AUC near 0.6), so the baseline was tuned before the comparison."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y, dtype=torch.float32)
    steps = EPOCHS * (len(Xt) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt) - BATCH + 1, BATCH):
            b = perm[i:i + BATCH]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    def score(Xn):
        with torch.no_grad():
            return model(torch.tensor((Xn - mu) / sd)).numpy()
    return score


# ---------------------------------------------------------------- metrics
def auc(score, y):
    """Area under the ROC curve via the Mann-Whitney statistic (ties counted half)."""
    pos, neg = score[y == 1], score[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc(score, y, n_pts=200):
    thr = np.quantile(score, np.linspace(0, 1, n_pts))[::-1]
    P, N = (y == 1).sum(), (y == 0).sum()
    tpr = [float(((score >= t) & (y == 1)).sum() / P) for t in thr]
    fpr = [float(((score >= t) & (y == 0)).sum() / N) for t in thr]
    return [0.0] + fpr + [1.0], [0.0] + tpr + [1.0]


def tpr_at(score, y, fpr=0.01):
    t = np.quantile(score[y == 0], 1 - fpr)
    return float((score[y == 1] > t).mean())


# ---------------------------------------------------------------- experiment
def main():
    t0 = time.time()
    rng_test = np.random.default_rng(123)
    cases = {"a_seen": ("E1", "A1"), "b_env_shift": ("E2", "A1"), "c_unseen_attack": ("E1", "A2"),
             "d_consistent_spoof": ("E1", "A3")}
    tests = {k: make_set(e, a, N_TEST, rng_test) for k, (e, a) in cases.items()}

    # sanity check of the physics fit on clean reports
    tau, anc, _ = make_set("E1", None, 2000, np.random.default_rng(7))
    p_hat, s_hat, _ = physics_fit(tau, anc)
    _, node, c = geometry(ENVS["E1"], 2000, np.random.default_rng(7))
    sanity = dict(median_position_error_m=float(np.median(np.linalg.norm(p_hat - node, axis=1))),
                  median_abs_sound_speed_error_mps=float(np.median(np.abs(1 / s_hat - c))))

    res = {"config": dict(K=K, sigma_ms=SIGMA * 1e3, p_multipath=P_MULTI, multipath_max_ms=MULTI_MAX * 1e3, envs=ENVS,
                          n_train=N_TRAIN, n_test=N_TEST, seeds=SEEDS, epochs=EPOCHS, mlp=f"3 x {WIDTH} SiLU, one-cycle Adam, lr {LR}",
                          layouts={k: layout(v).round(1).tolist() for k, v in ENVS.items()},
                          attacks=dict(A1="1 link +3..+15 ms", A2="2 links +-1.5..5 ms", A3="fake position 100-400 m")),
           "sanity_physics_fit_E1_clean": sanity, "physics": {}, "mlp": {k: [] for k in cases}, "roc": {}}

    for k, (tau, anc, y) in tests.items():
        _, _, score = physics_fit(tau, anc)
        res["physics"][k] = dict(auc=auc(score, y), tpr_at_1pct_fpr=tpr_at(score, y))
        res["roc"][k] = {"physics": roc(score, y)}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tau, anc, y = make_set("E1", "A1", N_TRAIN, rng)
        score_fn = train_mlp(features(tau, anc), y, seed)
        for k, (tau_t, anc_t, y_t) in tests.items():
            sc = score_fn(features(tau_t, anc_t))
            res["mlp"][k].append(dict(seed=seed, auc=auc(sc, y_t), tpr_at_1pct_fpr=tpr_at(sc, y_t)))
            if seed == 0:
                res["roc"][k]["mlp"] = roc(sc, y_t)
        print(f"seed {seed} done ({time.time() - t0:.0f} s)", flush=True)

    res["mlp_summary"] = {k: dict(auc=[float(np.mean([r["auc"] for r in v])), float(np.std([r["auc"] for r in v]))],
                                  tpr_at_1pct_fpr=[float(np.mean([r["tpr_at_1pct_fpr"] for r in v])),
                                                   float(np.std([r["tpr_at_1pct_fpr"] for r in v]))])
                          for k, v in res["mlp"].items()}
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "example2_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"sanity": sanity, "physics": res["physics"], "mlp": res["mlp_summary"],
                      "runtime_s": res["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
