# PIML-SAGSIN-Survey

Companion repository of the article

> M. Jouhari, H. Benaddi, and M. Guizani, "Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey," submitted to *IEEE Communications Surveys & Tutorials*, 2026.

The article reviews 102 papers (2019 to 2026) that embed a physical law in a learning model for the five segments of space-air-ground-sea integrated networks (SAGSIN): space, air, ground, sea surface, and underwater. This repository holds the screening table behind every count of the article, the code of the two worked examples, the decision framework in an editable form, and the supplementary material.

## Contents

| Path | Content |
|---|---|
| `data/screening_table.csv` | The 246 coded papers and resources: role, segment, layer or function, mechanism codes, data (simulated or measured), reported gain, DOI, and source of verification |
| `data/references.bib` | BibTeX file of the article |
| `code/colab/example1_acoustic_link_v2.ipynb` | Worked example 1: transmission loss of an underwater acoustic link (Section II-F) |
| `code/colab/example2_cross_segment.ipynb` | Worked example 2: seabed-to-satellite relay through a buoy (Section V-D) |
| `figures/decision_framework.drawio` | Decision framework of Fig. 8, editable in diagrams.net |
| `supplementary_material.md` | Selection details, full settings of the worked examples, abbreviations |
| `CITATION.cff` | Citation metadata |

## Inclusion rule

A paper is counted when it (i) appeared between 2019 and 2026 as a journal article, a conference paper, or an arXiv preprint, (ii) addresses a task in one of the five segments or at an interface between two segments, and (iii) embeds an identifiable physical law, that is, a governing equation, a propagation law, or a geometric or dynamic model, through one of the mechanisms below. Papers that use physics only as a generic input, as a system model, or as a schedule are listed in the screening table as PIML-adjacent and are not counted. The details are in Section I-D of the article and in Section S1 of the [supplementary material](supplementary_material.md).

## Taxonomy

Number of counted papers per segment and per category of use. Each paper is counted once, in the cell of its main task. A dash marks a cell without a counted paper.

| Segment | Physical layer | Link layer | Network layer | Digital twin | Security | Platform dynamics | Total |
|---|---|---|---|---|---|---|---|
| Space | 9 | – | 1 | 2 | 1 | 9 | 22 |
| Air | 5 | – | – | – | 2 | 4 | 11 |
| Ground | 20 | – | – | 4 | 1 | – | 25 |
| Sea surface | 5 | – | – | 1 | 1 | 4 | 11 |
| Underwater | 22 | 2 | 1 | 1 | – | 7 | 33 |
| Cross-segment | – | – | – | – | – | – | 0 |
| **Total** | 61 | 2 | 2 | 8 | 5 | 24 | **102** |

Papers per year (year of the cited version): 2019: 1, 2021: 1, 2022: 2, 2023: 8, 2024: 19, 2025: 31, 2026: 40. The count of 2026 covers the papers found up to 1 November 2026, the end of the search.

Of the 102 papers, 45 use measured data: 34 of the 78 papers on communication tasks and 11 of the 24 papers on platform dynamics.

## Mechanism codes

A paper can carry two codes, so the 102 papers have 122 labels.

| Code | Mechanism | How physics enters | Papers |
|---|---|---|---|
| L | Loss term | The residual of a governing equation is penalized at collocation points | 46 |
| A | Architecture or hard constraint | The structure of the network satisfies the law by construction | 27 |
| R | Hybrid residual | A physics model gives the main effect, and a network learns the correction | 11 |
| F | Physics features | Quantities computed by a law are given as inputs | 11 |
| S | Pretraining on simulations | A physics simulator generates the training data | 9 |
| G | Generative conditioning | Physical quantities condition or guide a generative model | 7 |
| O | Neural operator (additional tag) | An operator maps the configuration of the medium to the field | 5 |
| C | Consistency check | Observations are compared with a physical prediction | 4 |
| W | World model or RL constraint | A physical model predicts outcomes or removes infeasible actions | 2 |

## When and how to use physics-informed learning in SAGSIN

The decision framework of the article (Section IX-B, Fig. 8) starts from the conditions of deployment and not from a method. The four questions are answered in order.

| Question | Answer | Recommended model |
|---|---|---|
| Q1. Do the data cover the conditions of deployment? | Yes | Data-driven model; physics is optional, as input features (F) |
| | No | Go to Q2 |
| Q2. Is the form of the law trusted in this environment? | No | Calibrated law plus a learned correction (R), or a Gaussian process with the law as its mean; valid inside the data only; no physics loss (L) with this law |
| | Yes | Go to Q3 |
| Q3. Is the law a closed form or an ODE, or a PDE that needs a solver? | Closed form or ODE | Bounded least-squares fit of the law first; a residual (R) only if a structured residual remains; mask or cost (W) for decisions; residual test (C) for monitoring |
| | PDE | Go to Q4 |
| Q4. Is the environment known (profile, geometry, materials)? | Yes | Classical solver as the reference; surrogate (O, S, L) for many or real-time queries, within the family of its training environments |
| | No, sparse data | PINN inverse problem (L) or hard constraint (A), with the training effort reported |

The form of a law is tested by fitting its parameters and inspecting the residual of the fit: a residual at the level of the noise indicates a trusted form, and a residual with a structure in range, frequency, or season indicates a wrong form.

### Reporting checklist

Eight items are reported whatever the branch.

| Item | What is reported |
|---|---|
| R1 | Calibrated-law baseline |
| R2 | Data-only baseline with the same inputs |
| R3 | Ablation of the physical part |
| R4 | Split under a shift defined in physical terms (range, band, site, season, geometry) |
| R5 | At least five seeds, with their spread |
| R6 | Measured data, or the stated gap between simulation and measurement |
| R7 | Violation rate per episode (W) or test against a model-aware attacker (C) |
| R8 | Memory, latency, and energy on the target processor |

Table IX of the article applies the framework to twelve tasks, from LEO channel prediction to spoofing detection, with the failure mode recorded for each of them. The framework is derived from the corpus and from two simulated examples, so it is a working hypothesis for the design of experiments and not a validated procedure.

## Worked examples

Both notebooks run on a central processing unit, in Google Colab or in a local Jupyter installation.

| Example | Question | Main result |
|---|---|---|
| 1. Underwater acoustic link | Which model predicts the transmission loss at ranges and frequencies without measurement? | With a law of the right form, a least-squares fit of three parameters extrapolates as well as a PINN, and a plain network becomes worse than the textbook formula. With a law of the wrong form, only a correction learned from data removes the bias, and only inside the data. |
| 2. Seabed-to-satellite relay | Which buoy should relay a seabed sensor to a LEO satellite, and at which rate, in conditions that were not measured? | Calibrated acoustic and satellite laws joined at the buoy predict the end-to-end rate under shift better than networks trained on the same data, and the model of the gateway is the largest source of error. |

Example 1 takes about 40 min on a free Colab instance. The settings of both examples are listed in Sections S2 and S3 of the [supplementary material](supplementary_material.md).

Both examples use simulated channels with known laws. They show how the mechanisms behave under a controlled shift, and they do not replace a test on measured data.

## Screening table

Each row of `data/screening_table.csv` is one paper or resource.

| Column | Meaning |
|---|---|
| BibTeX key | Key of the entry in `data/references.bib` |
| Role | PIML primary (counted), PIML-adjacent, data-driven baseline, classic model, dataset or simulator, survey or background, excluded |
| Segment | Space, air, ground, sea surface, underwater, cross-segment, or general |
| Layer / function | Physical layer, link, network, digital twin, security, platform dynamics, or method |
| How physics enters | Mechanism codes of the paper |
| Physics used | Law or model that the paper embeds |
| Data | Simulated, measured, or both, with the dataset when it is named |
| Reported gain | Main figure reported by the authors |
| DOI, Verified via | Identifier and the source against which the metadata were checked (Crossref, arXiv, or the page of the publisher) |

## Contributing

A paper that embeds a physical law for a SAGSIN task and is missing from the table can be proposed through an issue or a pull request. A proposal gives the DOI or the arXiv identifier, the segment, the layer or function, the mechanism codes, and whether measured data are used. Negative results and ablations of the physical part are of particular interest, since they are rare in the corpus.

## Citation

```bibtex
@article{jouhari2026piml,
  author  = {Jouhari, Mohammed and Benaddi, Hafsa and Guizani, Mohsen},
  title   = {Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey},
  journal = {Submitted to IEEE Communications Surveys \& Tutorials},
  year    = {2026}
}
```

## Contact

Mohammed Jouhari, Department of Computer Science, Ibn Tofail University, Kenitra, Morocco (ORCID: 0000-0001-5406-8594). Questions and corrections are welcome through the issue tracker.
