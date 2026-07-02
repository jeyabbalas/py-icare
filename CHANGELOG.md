# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-07-01

A fit-once / apply-many entry point so a single reusable model can score many covariate profile batches
without re-reading the reference dataset (the efficiency lever for large or streamed datasets, e.g. from
the `wasm-icare` JavaScript SDK). Purely additive; default behavior is unchanged and all existing tests
pass.

### Added

- `build_absolute_risk_model(...)`: builds a reusable `AbsoluteRiskModel` once — reading the reference
  dataset a single time and fitting the population distribution, betas, and baseline & competing hazards —
  and returns it. Accepts the same model arguments (and the same in-memory objects) as
  `compute_absolute_risk`.
- `AbsoluteRiskModel.apply_to_profile(apply_age_start, apply_age_interval_length,
  apply_covariate_profile_path, ...)`: applies a built model to a batch of covariate profiles, reusing the
  fitted state (the reference is not re-read), and returns results packaged identically to
  `compute_absolute_risk`. Supports the general-purpose covariate model; the SNP option remains available
  through `compute_absolute_risk`.

## [1.2.0] - 2026-07-01

Backward-compatible in-memory I/O so the library can be driven without touching disk (for example from
the `wasm-icare` JavaScript SDK under Pyodide). Default behavior is unchanged; all existing tests pass.

### Added

- In-memory inputs for every `*_path` argument of `compute_absolute_risk`,
  `compute_absolute_risk_split_interval`, and `validate_absolute_risk_model` (including the `*_path`
  values nested inside `icare_model_parameters`): pass a pandas DataFrame for tabular data, a dict for the
  log-odds-ratio arguments, or an inline Patsy formula string for the covariate-formula arguments, in
  place of a file path.
- An `output_format` argument (default `'json'`, or `'dataframe'`) on all three public functions. In
  `'dataframe'` mode, `profile` / `study_data` / `incidence_rates` / `category_specific_calibration` are
  returned as pandas DataFrames and each `reference_risks` interval's `population_risks` as a NumPy array,
  instead of the records-oriented JSON strings.

### Changed

- The split-interval combiner is now DataFrame-native: it no longer serializes each sub-interval to JSON
  and re-parses it before combining. Results are unchanged to within test tolerance (the removed round
  trip only dropped an intermediate 10-digit rounding); the default JSON output normalizes whole-number
  covariate tokens (e.g. `0` to `0.0`), consistent with the single-interval output.

## [1.1.0] - 2026-07-01

The first release since 1.0.0: a full modernization of the dependency stack, new
model-validation metrics, and a corrected AUC.

### Added

- Overall Brier score in the `validate_absolute_risk_model` output, reported with
  its variance and 95% confidence interval (`brier_score`).
- Category-specific (per-bin) Expected/Observed ratios in the calibration output
  (`expected_by_observed_ratio`), alongside the overall ratio.

### Changed

- Rebuilt on the current scientific-Python stack: NumPy >= 2.4.3, pandas >= 3.0.2,
  SciPy >= 1.18.0, and patsy >= 1.0.2. All dependencies remain Pyodide / WebAssembly
  compatible.
- Performance: vectorized the AUC calculation via broadcasted ranking; optimized
  absolute-risk estimation (model-free imputation, the risk score hoisted in
  baseline-hazard estimation, and the linear predictor computed once).
- Packaging migrated to a single-source version in `pyproject.toml`, with a pytest
  suite validated against R-iCARE reference outputs and automated PyPI publishing
  via GitHub Actions (Trusted Publishing / OIDC).

### Removed

- Support for Python 3.11 and earlier. Python >= 3.12 is now required (SciPy >= 1.18
  dropped 3.11).

### Fixed

- Corrected AUC for tied risk scores. Ties were previously mishandled; AUC is now
  computed correctly using a tie tolerance. This is an intentional, documented
  deviation from R-iCARE, which handles ties incorrectly.
- Robust Hosmer-Lemeshow binning: unweighted binning no longer crashes on sparse or
  empty bins, and calibration statistics are derived from the realized bin counts.
- pandas 3.0 compatibility: fixed the `Categorical` assignment and `pd.read_json`
  breakages introduced by pandas 3.

### Notes

Under pandas 3.0, `pd.read_json` no longer accepts a raw JSON string. Wrap the
returned strings in `io.StringIO` when reading Py-iCARE's JSON output:

```python
import io, pandas as pd
profile = pd.read_json(io.StringIO(result["profile"]), orient="records")
```

## [1.0.0] - 2023-05-25

- Initial public release of Py-iCARE on PyPI.

[1.2.0]: https://pypi.org/project/pyicare/1.2.0/
[1.1.0]: https://pypi.org/project/pyicare/1.1.0/
[1.0.0]: https://pypi.org/project/pyicare/1.0.0/
