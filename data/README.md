# Data

| File | Content |
|---|---|
| `screening_table.csv` | The 246 papers and resources coded for the survey, one per row (UTF-8, comma-separated) |
| `references.bib` | BibTeX file of the article |

## Columns of the screening table

| Column | Meaning |
|---|---|
| BibTeX key | Key of the entry; it is the key of `references.bib` when the last-but-one column is `yes` |
| Role | PIML primary, PIML-adjacent, PIML method (general), data-driven baseline, classic model, dataset/simulator, survey/background, excluded (venue rule), or conference version |
| Counted in the corpus | `yes` for the 102 papers counted in the article |
| Segment | Space, air, ground, sea surface, underwater, cross-segment, or general |
| Layer / function | Physical layer, link layer, network layer, digital twin, security, platform dynamics, method, or system |
| Mechanism codes | How physics enters the model (codes below); empty when no law is embedded |
| Physics used | Law, model, or simulator that the paper relies on |
| Data | Simulated, measured, or both, with the dataset when it is named; `n.s.` when not stated |
| Measured data | For counted papers: `yes` when part of the training or test data was measured |
| Reported gain | Main figure reported by the authors; `n.s.` when the abstract states none |
| Year, First author, Title, Venue, Peer-reviewed, DOI | Bibliographic record |
| In references.bib | `yes` when the article cites the entry |
| Notes | Remarks on border cases |

## Mechanism codes

| Code | Mechanism | Counted papers |
|---|---|---|
| L | Loss term: the residual of a law is penalized | 46 |
| A | Architecture or hard constraint: the structure satisfies the law | 27 |
| R | Hybrid residual: a physics model plus a learned correction | 11 |
| F | Physics features: quantities computed by a law are inputs | 11 |
| S | Pretraining on simulations | 9 |
| G | Generative conditioning | 7 |
| O | Neural operator (additional tag) | 5 |
| C | Consistency check | 4 |
| W | World model or reinforcement-learning constraint | 2 |

A paper can carry two codes, so the 102 counted papers have 122 labels.

## Counts that the table reproduces

| Quantity of the article | Filter on the table | Value |
|---|---|---|
| Coded papers and resources | all rows | 246 |
| Corpus | `Counted in the corpus = yes` | 102 |
| Papers on communication tasks | counted, `Layer / function` other than platform dynamics | 78 |
| Papers on platform dynamics | counted, `Layer / function = Platform dynamics` | 24 |
| Papers with measured data | counted, `Measured data = yes` | 45 (34 on communication tasks, 11 on platform dynamics) |
| Papers removed by the venue rule | `Role = Excluded (venue rule)` | 3 |
| Papers per year | counted, grouped by `Year` | 2019: 1, 2021: 1, 2022: 2, 2023: 8, 2024: 19, 2025: 31, 2026: 40 |

Grouping the counted rows by `Segment` and `Layer / function` gives the taxonomy of Fig. 2 of the article.

```python
import pandas as pd

t = pd.read_csv("screening_table.csv")
corpus = t[t["Counted in the corpus"] == "yes"]
print(len(t), len(corpus))
print(pd.crosstab(corpus["Segment"], corpus["Layer / function"]))
print(corpus["Mechanism codes"].str.split(", ").explode().value_counts())
```

## Notes

- A paper counts as using measured data when part of its training or test data was measured, including measured inputs such as two-line element records or vessel tracks, and not when a simulation only takes real terrain, weather, or ephemerides as input.
- The year, the venue, and the title of the entries cited in the article are those of `references.bib`, that is, of the cited version.
- Fifty rows are background material that the article does not cite; their bibliographic record is given in the table only.
