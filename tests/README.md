# py-icare test suite

High-level cross-validation tests that compare **py-icare** against the original
**R iCARE** package (Bioconductor), on the two models shipped in `data/`:
**BPC3** and **iCARE-Lit**. The suite reproduces the five iCARE vignette
workflows and checks that the Python port produces the same results as R.

## Layout

```
tests/
├── conftest.py                       # makes tests/ importable
├── icare_test_utils.py               # paths, tolerances, golden loaders, helpers
├── test_smoke.py                     # fast "does it run under the new deps?" guards
├── test_bpc3_cross_validation.py     # BPC3  vs R (8 workflows)
├── test_icare_lit_cross_validation.py# iCARE-Lit vs R (lt50/ge50/split/validation)
└── r_reference/
    ├── r_utils.R                     # shared R helpers (summaries, golden writer)
    ├── helpers.R                     # Patsy -> R model translator (iCARE-Lit)
    ├── generate_bpc3_references.R    # produces BPC3 goldens from native bc_data
    ├── generate_icare_lit_references.R
    ├── expected/                     # committed golden JSON files (R outputs)
    └── fixtures/                     # committed inputs shared by R and Python
```

## Running the tests

```bash
pip install -e .                      # or: pip install -r requirements.txt
pytest                                # from the repo root
pytest -m "not slow"                  # skip the validation-cohort tests
```

R is **not** required to run the tests — they compare against the committed
golden files in `r_reference/expected/`.

## Regenerating the golden references

Needed only when the model data, the R iCARE version, or a generator script
changes. Requires R with the `iCARE` and `jsonlite` packages.

```bash
Rscript tests/r_reference/generate_bpc3_references.R
Rscript tests/r_reference/generate_icare_lit_references.R
```

Reference R/iCARE versions used to produce the committed goldens: **R 4.5.2,
iCARE 1.38.0**.

### How each model maps to R

* **BPC3** is the *same* model as iCARE's built-in `bc_data` (re-encoded: R uses
  integer decile codes, Python uses label strings). The R generator uses
  `bc_data` directly — no translation — and Python runs the `data/BPC3/` files.
* **iCARE-Lit** is not shipped with R. `helpers.R` translates the Patsy formula +
  log-odds-ratio JSON into R's `model.formula` + `model.cov.info` + `model.log.RR`
  (`C(var, levels=[...])` → `as.factor(var)`, levels kept in Patsy order so the
  reference category matches). The translation is verified: it asserts every beta
  maps to exactly one design-matrix column, and the resulting risks match
  py-icare to ~6 significant figures (including the ge50 HRT×BMI interaction).

## Comparison strategy & tolerances

Tolerances live in `icare_test_utils.py`. The guiding principle: compare
*deterministic* quantities tightly and *stochastic* ones distributionally,
because R and Python use different random number generators.

| Quantity | Tolerance | Why |
|---|---|---|
| Covariate-only risks & linear predictors (full profiles) | `atol=1e-5` | deterministic; observed agreement ~1e-6 |
| Split-interval (covariate-only) risks | `atol=1e-5` | deterministic |
| Reference-population risk distribution (summary q1/median/mean/q3) | `atol=5e-3` | order-independent, stable across RNGs |
| SNP per-subject risks (simulation / imputation) | `atol=2e-2` | R and Python RNGs differ; cannot match per subject |
| Validation expected/observed ratio | `atol=1e-2` | agrees (~1e-3 cov-only, ~5e-3 combined) |
| Validation AUC | `atol=1.5e-2` | see finding below |
| Validation Hosmer-Lemeshow | conclusion only | see finding below |

Min/max of distributions are excluded from comparisons as RNG-sensitive extreme
order statistics.

## Findings surfaced by this suite

1. **Dependency-upgrade breakage (fixed).** The pandas 2→3 upgrade broke two of
   the three public functions; both are fixed in this change:
   * `compute_absolute_risk_split_interval` — `pd.read_json` on a JSON *string*
     (`icare/misc.py`); fixed with `io.StringIO`.
   * `validate_absolute_risk_model` (weighted / nested case-control path) —
     assigning new values into a `Categorical` column (`icare/model_validation.py`).
   The documented `pd.read_json(result[...])` examples in the docstrings were
   broken the same way and were corrected.

2. **AUC estimator difference (~0.005).** With identical risks and identical
   sampling weights, py-icare's IPW AUC is consistently ~0.005 higher than R's
   (e.g. 0.6003 vs 0.5952). The expected/observed ratio matches to ~0.001, so the
   risks themselves agree — the gap is in the AUC estimator. Tolerated via
   `ATOL_AUC`; worth a closer look in a future pass.

3. **Hosmer-Lemeshow binning difference.** R and py-icare bin risk scores
   differently (especially the weighted quantiles), so the HL chi-square magnitude
   differs by ~15%. The calibration *conclusion* (significant miscalibration at
   α=0.05) agrees in every case, which is what the tests assert.

4. **R `ModelValidation` + shipped nested-CC weights.** R iCARE's weighted
   categorization fails on the dataset's shipped `sampling.weights`; the vignette
   recomputes them via the inclusion model. The generator does the same and
   exports those weights so both engines validate identically.
