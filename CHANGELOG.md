# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.1.0]: https://pypi.org/project/pyicare/1.1.0/
[1.0.0]: https://pypi.org/project/pyicare/1.0.0/
