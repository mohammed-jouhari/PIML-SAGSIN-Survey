"""Build README.md from data/screening_table.csv.

Run from the repository root after any change to the screening table:
    python scripts/build_readme.py
The paper lists, the taxonomy counts and all section sizes are generated; edit the text blocks below for prose.
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "screening_table.csv").fillna("")
df["Year"] = df["Year"].astype(int)

SEGS = ["Space", "Air", "Ground", "Sea surface", "Underwater", "Cross-segment"]
LAYS = [("PHY", "Physical layer"), ("Link", "Link layer"), ("Network", "Network layer"),
        ("Digital twin", "Digital twin"), ("Security", "Security"), ("Platform dynamics", "Platform dynamics")]
CODE = {"loss": "L", "architecture": "A", "residual": "R", "generative-conditioning": "G", "operator": "O",
        "world-model/RL-constraint": "W", "consistency-check": "C", "features": "F", "sim-pretraining": "S"}
REPO = "https://github.com/mohammed-jouhari/PIML-SAGSIN-Survey"


def anchor(text):
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).replace(" ", "-")


def link(doi):
    m = re.match(r"10\.48550/arxiv\.(.+)", doi, re.I)
    return f"https://arxiv.org/abs/{m.group(1)}" if m else f"https://doi.org/{doi}"


def codes(mech):
    return ", ".join(sorted({CODE[p.strip()] for p in str(mech).split(",") if p.strip() in CODE}))


def item(r, detail=True):
    venue = r["Venue"].replace("|", "/")
    s = f"- **{r['Title']}**. {r['First author']} *et al.*, *{venue}*, {r['Year']}. [[link]]({link(r['DOI'])})"
    if r["Peer-reviewed"].startswith("No"):
        s += " `preprint`"
    if detail:
        bits = []
        c = codes(r["How physics enters"])
        if c:
            bits.append(f"Mechanism: **{c}**")
        if r["Physics used"] and r["Physics used"] not in ("n/a",):
            bits.append(f"Physics: {r['Physics used']}")
        if r["Data (simulated / measured)"] and r["Data (simulated / measured)"] != "n/a":
            bits.append(f"Data: {r['Data (simulated / measured)']}")
        if r["Reported gain (abstract)"] and r["Reported gain (abstract)"] not in ("n/a", "n.s."):
            bits.append(f"Reported: {r['Reported gain (abstract)']}")
        if bits:
            s += "<br>  " + " · ".join(bits)
    return s


def sort(d):
    return d.sort_values(["Year", "First author"], ascending=[False, True])


prim = df[df["Role"] == "PIML primary"]
cnt = Counter(zip(prim["Segment"], prim["Layer / function"]))
years = Counter(prim["Year"])
L = []
L += [
    "# PIML-SAGSIN-Survey: Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks",
    "",
    "[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)",
    "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)",
    f"![Papers](https://img.shields.io/badge/PIML%20papers-{len(prim)}-blue)",
    "",
    "Companion repository of a tutorial and survey. A curated list of **physics-informed machine learning (PIML)** for **space-air-ground-sea integrated networks (SAGSIN)**:",
    "satellites and non-terrestrial networks, UAVs and HAPS, terrestrial radio, maritime links and underwater acoustic and optical networks.",
    "",
    "This repository accompanies the article",
    "",
    "> M. Jouhari, \"Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey,\" 2026. *(arXiv link will be added here.)*",
    "",
    "It contains the list of surveyed papers organized by the taxonomy of the article (segment × layer × how physics enters),",
    "the full screening table, the verified BibTeX file, and the code of the two worked examples.",
    "",
    f"**Inclusion rule.** A paper is listed as PIML work only if it embeds an identifiable physical law (a governing equation, a propagation law, or a geometric or dynamic model) in its features, loss, architecture or decision process. Related work that uses physical knowledge without embedding a law is listed separately. Every entry was checked against Crossref or arXiv.",
    "",
    "## Contents",
    "",
    "- [Taxonomy](#taxonomy)",
    "- [How physics enters: mechanism codes](#how-physics-enters-mechanism-codes)",
]
for s in SEGS[:-1]:
    L.append(f"- [{s}](#{anchor(s)})")
L += ["- [Cross-segment and interfaces (open)](#cross-segment-and-interfaces-open)",
      "- [Related work without an embedded physical law](#related-work-without-an-embedded-physical-law)",
      "- [PIML methods (general)](#piml-methods-general)",
      "- [Datasets, simulators and benchmarks](#datasets-simulators-and-benchmarks)",
      "- [Data-driven baselines and classic models](#data-driven-baselines-and-classic-models)",
      "- [Surveys and background](#surveys-and-background)",
      "- [Code of the worked examples](#code-of-the-worked-examples)",
      "- [Repository layout](#repository-layout)",
      "- [Contributing](#contributing)",
      "- [Citation](#citation)", ""]

L += ["## Taxonomy", "",
      "Number of PIML papers per SAGSIN segment (rows) and layer or function (columns). Empty cells are open research problems.", "",
      "![Taxonomy of PIML for SAGSIN](figures/taxonomy.png)", "",
      "| Segment | " + " | ".join(lab for _, lab in LAYS) + " | Total |",
      "|---|" + "---|" * (len(LAYS) + 1)]
for s in SEGS:
    row = [cnt.get((s, l), 0) for l, _ in LAYS]
    cells = [str(n) if n else "open" for n in row]
    L.append(f"| {s} | " + " | ".join(cells) + f" | {sum(row)} |")
L.append(f"| **Total** | " + " | ".join(str(sum(cnt.get((s, l), 0) for s in SEGS)) for l, _ in LAYS) + f" | **{len(prim)}** |")
L += ["", "Papers per year: " + ", ".join(f"{y}: {years[y]}" for y in sorted(years)) + ".", "",
      "![PIML papers per year and segment](figures/trend.png)", ""]

L += ["## How physics enters: mechanism codes", "",
      "| Code | Mechanism | How physics enters |", "|---|---|---|",
      "| F | Physics features | Quantities computed by a law are given as inputs |",
      "| L | Loss term | Residual of a governing equation is penalized (PINN) |",
      "| A | Architecture or hard constraint | The network structure satisfies the law by construction |",
      "| R | Hybrid residual | A physics model gives the main effect, a network learns the correction |",
      "| G | Generative conditioning | Physical quantities condition or guide a generative model |",
      "| O | Neural operator | An operator maps environments (e.g., sound-speed profiles) to fields |",
      "| W | World model or RL constraint | A physical model predicts outcomes or masks infeasible actions |",
      "| C | Consistency check | Observations are compared with a physical prediction (security, anomalies) |",
      "| S | Simulation pretraining | A physics simulator generates training data |", ""]

for s in SEGS[:-1]:
    L += [f"## {s}", ""]
    sub = prim[prim["Segment"] == s]
    for l, lab in LAYS:
        d = sub[sub["Layer / function"] == l]
        if len(d) == 0:
            continue
        L += [f"### {s}: {lab} ({len(d)})", ""]
        L += [item(r) for _, r in sort(d).iterrows()] + [""]

L += ["## Cross-segment and interfaces (open)", "",
      "No surveyed paper learns across two segments or models an interface (slant atmospheric paths, line-of-sight transitions, sea-surface ducts, the water-air boundary). "
      "If you know of one, please open an issue or a pull request.", ""]

sections = [("Related work without an embedded physical law", "PIML-adjacent", True),
            ("PIML methods (general)", "PIML method (general)", True),
            ("Datasets, simulators and benchmarks", "Dataset/simulator", True),
            ("Data-driven baselines and classic models", ["Data-driven baseline", "Classic model"], True),
            ("Surveys and background", "Survey/background", False)]
for title, roles, det in sections:
    roles = roles if isinstance(roles, list) else [roles]
    d = df[df["Role"].isin(roles)]
    L += [f"## {title}", "", f"{len(d)} entries.", ""]
    L += [item(r, det) for _, r in sort(d).iterrows()] + [""]

L += ["## Code of the worked examples", "",
      "The `code/` folder reproduces both worked examples of the article on a laptop CPU (no GPU).", "",
      "| Example | Script | What it shows | Run time (2-thread CPU) |", "|---|---|---|---|",
      "| 1, Section II | `example1_acoustic_link.py` | MLP vs PINN vs hybrid residual model for the transmission loss of an underwater acoustic link, inside and outside the training region | about 11 min |",
      "| 2, Section VII | `example2_spoofing.py` | Physics-residual detector vs MLP detector of spoofed delay reports in an acoustic network, with ROC curves | about 7 min |", "",
      "```bash", "cd code", "pip install -r requirements.txt", "python run_all.py            # both examples, then the figures",
      "python run_all.py --figures  # only redraw the figures from results/*.json", "```", "",
      "| Example 1: RMSE vs training samples | Example 2: ROC curves |", "|---|---|",
      "| ![Example 1](figures/example1_rmse.png) | ![Example 2](figures/example2_roc.png) |", "",
      "## Repository layout", "",
      "```",
      "data/screening_table.xlsx   full screening table (role, segment, layer, physics type, mechanism, data, reported gain, DOI)",
      "data/screening_table.csv    the same table as CSV (source of this README)",
      "data/references.bib         verified BibTeX of every entry",
      "code/                       worked examples, results (JSON) and figure script",
      "scripts/build_readme.py     regenerates this README from the CSV",
      "scripts/make_taxonomy_figure.py  regenerates the TikZ taxonomy figure from the table",
      "figures/                    figures used in this README",
      "```", "",
      "## Contributing", "",
      "Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). In short: add one row to `data/screening_table.csv` "
      "with a DOI or arXiv identifier, the segment, the layer and how physics enters, then run `python scripts/build_readme.py`.", "",
      "## Citation", "",
      "If this list or the code helps your work, please cite the article (the entry will be updated with the arXiv identifier):", "",
      "```bibtex",
      "@misc{jouhari2026piml,",
      "  author = {Jouhari, Mohammed},",
      "  title  = {Physics-Informed Machine Learning for Space-Air-Ground-Sea Integrated Networks: A Tutorial and Survey},",
      "  year   = {2026},",
      "  note   = {Preprint}",
      "}",
      "```", "",
      f"*Generated by `scripts/build_readme.py` from the screening table. Last update: 2026-10-15.*", ""]
(ROOT / "README.md").write_text("\n".join(L), encoding="utf-8")
print("README written:", len(prim), "PIML papers,", len(df), "entries")
