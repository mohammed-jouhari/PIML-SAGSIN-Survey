# Contributing

Thank you for helping keep this list complete and correct.

## Add a paper

1. Check that the paper **embeds an identifiable physical law** (a governing equation, a propagation law, or a
   geometric or dynamic model) in its features, loss, architecture or decision process, and that it concerns a
   SAGSIN segment (space, air, ground, sea surface, underwater) or an interface between segments.
   Papers that use physical knowledge without embedding a law are welcome too, with the role `PIML-adjacent`.
2. Add one row to `data/screening_table.csv`. Required fields:
   - `BibTeX key`, `Title`, `First author`, `Venue`, `Year`, `DOI` (or `10.48550/arXiv.<id>` for preprints);
   - `Role`: `PIML primary` or `PIML-adjacent`;
   - `Segment`: Space, Air, Ground, Sea surface, Underwater or Cross-segment;
   - `Layer / function`: PHY, Link, Network, Digital twin, Security or Platform dynamics;
   - `How physics enters`: one or more of `features, loss, architecture, residual, generative-conditioning,
     operator, world-model/RL-constraint, consistency-check, sim-pretraining`;
   - `Physics used`, `Data (simulated / measured)` and `Reported gain (abstract)` as stated by the paper.
3. Add the BibTeX entry to `data/references.bib`, taken from the publisher or from doi.org.
4. Run `python scripts/build_readme.py` and open a pull request.

## Report an error

Open an issue with the paper and the field that is wrong. Classifications are based on the abstracts, so
corrections from authors are especially welcome.
