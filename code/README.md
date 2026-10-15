# Code for "Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey"

This folder reproduces every simulation number and figure in the article. All results in the
text come from the JSON files written by these scripts.

## Contents

| File | Purpose |
|---|---|
| `physics.py` | Thorp absorption, Stojanovic attenuation and ambient noise (Section III, Eqs. for the underwater segment) |
| `example1_acoustic_link.py` | Worked example of Section II: MLP vs PINN vs hybrid residual model for the transmission loss of an underwater acoustic link |
| `example2_spoofing.py` | Worked example of Section VII: physics-residual detector vs MLP detector of spoofed delay reports in an underwater acoustic network |
| `make_figures.py` | Builds the Example 1 figures, `figures/fig_example2_roc.pdf` and the trend figure `figures/fig_trend.pdf` (from `../data/screening_table.xlsx`) |
| `run_all.py` | Runs all examples, then makes the figures |
| `results/` | JSON outputs (created by the scripts) |

## Install and run

```bash
pip install -r requirements.txt
python run_all.py            # about 18 minutes on a 2-thread CPU (both examples)
python run_all.py --figures  # only redraw the figures from existing results
```

A CPU is enough; no GPU is used. Seeds are fixed. Small floating-point differences between
PyTorch versions or CPUs can change the last digit of some values.

## Example 1 in brief

- True channel: `TL(l,f) = 10 k log10(l / 1 m) + gamma * l * Thorp(f) + c0 + ripple(l)` with
  k = 1.7, gamma = 1.25, c0 = 3 dB and a 2 dB range-dependent ripple that no model contains.
- Nominal physics: k = 1.5, gamma = 1, no offset, no ripple.
- Training data: N in {10, 30, 100, 300} noisy samples (sigma = 1.5 dB) with l <= 5 km and f <= 25 kHz.
- Test: noise-free channel on 4000 random points of l in [0.2, 10] km and f in [2, 40] kHz,
  split into inside (interpolation) and outside (extrapolation) the training region.
- Models share one network (2 x 64, tanh, inputs log-distance and frequency):
  - MLP: data loss only; Adam, 3000 steps.
  - PINN: data loss + residuals of the distance and frequency derivatives of the model at
    1000 collocation points over the full domain; k and gamma are learned. Adam (2000 steps),
    then L-BFGS (up to 400 iterations).
  - Hybrid: nominal physics + network correction; Adam, 3000 steps.
  L-BFGS is not used for the MLP and the hybrid model because it made them unstable on sparse data.
- A sanity check (dense, noise-free data, no ripple) confirms that the PINN recovers k and gamma.
- 5 seeds per configuration.

## Example 2 in brief

- A node reports one-way delays to 6 anchors (4 surface buoys, 2 seabed nodes). Noise 0.5 ms, multipath excess delay
  (probability 0.05, up to 2 ms).
- Deployments: E1 (2 km x 2 km, 200 m deep, c in [1490, 1510] m/s, used for training) and E2 (4 km x 4 km, 1000 m deep,
  c in [1515, 1535] m/s, new deployment). One fixed anchor layout per deployment.
- Attacks: A1 one link +3..15 ms (seen in training), A2 two links +-1.5..5 ms, A3 all delays from a fake position
  100-400 m away (physics-consistent).
- Physics detector: Gauss-Newton fit of position and slowness; score = residual sum of squares / sigma^2. No training.
- MLP detector: 6 delays in, 3 x 256 SiLU, one-cycle Adam (lr 3e-3), 150 epochs, 50,000 training reports in E1, 5 seeds.
- Metrics: AUC and detection rate at 1% false alarms on 4,000 test reports per case.

## Mapping to the article

Every simulation number in Sections II-F and VII-E of the article is read from `results/example1_results.json`
and `results/example2_results.json`. The JSON files in this folder are the ones used for the article.

## License

MIT (see `../LICENSE`).
