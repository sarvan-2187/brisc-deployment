# Paper 2141 analysis scripts

Offline analysis for the BRISC-Edge revision (ICMLDE 5.0). Scripts expect the original layout
(`<root>/brain-tumor/refactor-kit/analysis/`, with `<root>/brain-tumor-deploy/` beside it);
adjust `ROOT`/`DEPLOY` in `make_paper_numbers.py` to run from this folder.

- `make_paper_numbers.py`: E2 paired statistics, de-duplicated subset, tau sweep, ECE, error figure
- `routed_variant.py`: per-class Bonferroni intervals and the offline class-routed variant
- `out/`: generated numbers; `results/near_duplicate_pairs.csv`: perceptual-hash audit
- E2 per-slice results: `../eval_results/per_image_results.csv`
- Pi logs (per-inference CSV, setup and summary per run): `../results/pi-rerun-2026-10/`
  (E6 sessions: `e6_college/` = room A, `e6/` = room B; excluded runs listed in `e3/INCIDENT.txt`)
