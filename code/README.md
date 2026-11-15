# Code of the worked examples

Two simulated examples support the article. Each one exists as a Python script and as a Jupyter notebook with the same content.

| Example | Section of the article | Script | Notebook | Libraries |
|---|---|---|---|---|
| 1. Underwater acoustic link | II-F | `example1_acoustic_link.py` | `notebooks/example1_acoustic_link.ipynb` | NumPy, SciPy, PyTorch, Matplotlib |
| 2. Seabed-to-satellite relay through a buoy | V-D | `example2_cross_segment.py` | `notebooks/example2_cross_segment.ipynb` | NumPy, SciPy, scikit-learn, Matplotlib |

## Run

```bash
pip install -r requirements.txt

# full runs (5 seeds, the setting of the article)
python example1_acoustic_link.py
python example2_cross_segment.py

# quick runs (1 seed)
NB_MODE=quick python example1_acoustic_link.py
NB_MODE=quick python example2_cross_segment.py
```

On Windows PowerShell, the quick mode is set with `$env:NB_MODE = "quick"` before the command. The notebooks run as they are in Google Colab, where all libraries are preinstalled; the variable `MODE` of the first code cell selects the full or the quick run.

A full run takes about 40 min for example 1 and 12 to 18 min for example 2 on a free Colab CPU.

## Outputs

Each run writes into the folder from which it is started.

| File | Content |
|---|---|
| `results/example1_results_<mode>.json` | Errors of every model, ocean, training-set size, and seed; paired comparisons; ablation of the GP length scale |
| `results/example2_results_<mode>.json` | Errors and throughput losses of every model, gateway, training-set size, and seed; fitted parameters |
| `figures/fig_example1.pdf` | Fig. 4 of the article |
| `figures/fig_example2_setup.pdf` | Fig. 5 of the article |
| `figures/fig_example2_practical.pdf` | Fig. 6 of the article |
| `figures/fig_example2_halfduplex.pdf`, `figures/fig_example2_fullduplex.pdf` | Same layout as Fig. 6 for the two ideal gateways |
| `figures/fig_example2_calibration.pdf` | Bounded against unbounded calibration |

The folder `results/` of this repository holds the JSON files of a full run of each example (torch 2.15.0, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, two CPU threads). The run of example 2 gives the numbers of Section V-D of the article. The run of example 1 gives the numbers of Section II-F in 50 of the 52 rows of its check; the two other rows are the errors of the PINN outside the measured region at 300 samples (1.2 dB against 1.0 dB in ocean A, and 7.6 dB against 8.1 dB in ocean B), for the reason given below.

## Reproducibility

- Every random draw is seeded: seed `s` fixes the training set, its noise, and the initial weights of repetition `s`, and the test sets have fixed seeds.
- Example 2 uses NumPy, SciPy, and scikit-learn only, and its numbers do not depend on the machine beyond rounding.
- In example 1, the L-BFGS stage of the PINN amplifies floating-point rounding, so the mean errors of the PINN can differ by up to about 20% between processors. The numbers of the article come from a run on a free Colab CPU. Section 10 of the script compares each number quoted in Section II-F of the article with the run and flags the rows that differ by more than 5%.

## Notebooks

The notebooks are generated from the scripts, which use the percent format (`# %%` for a code cell, `# %% [markdown]` for a text cell). After a change to a script, `python build_notebooks.py` rebuilds both notebooks.
