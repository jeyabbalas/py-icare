"""Shared helpers and constants for the py-icare cross-validation test suite.

The suite compares py-icare's output against golden reference files produced by
the original R iCARE package (see ``tests/r_reference/``). Helpers here load
those golden files, parse py-icare's JSON-string results, and summarize risk
distributions the same way the R generator does so the two sides are
comparable.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- Locations -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BPC3_DIR = DATA_DIR / "BPC3"
ICARE_LIT_DIR = DATA_DIR / "iCARE-Lit"
R_REFERENCE_DIR = Path(__file__).resolve().parent / "r_reference"
EXPECTED_DIR = R_REFERENCE_DIR / "expected"
FIXTURES_DIR = R_REFERENCE_DIR / "fixtures"

# Seed passed to py-icare wherever SNP simulation/imputation is involved.
GOLDEN_SEED = 50

# --- Tolerances (rationale documented in tests/README.md) ------------------
# Deterministic models (covariate-only, full profiles): py matches R to ~1e-6,
# so 1e-5 is a tight-but-safe ceiling that still catches real regressions.
ATOL_DETERMINISTIC = 1e-5
# Summary statistics of large populations: stable across R/Python RNGs.
ATOL_DISTRIBUTION = 5e-3
# Per-subject risks that depend on SNP imputation/simulation. R and Python use
# different RNGs, so individual values cannot match exactly -- only loosely.
ATOL_STOCHASTIC = 2e-2
# Validation metrics.
ATOL_EO = 1e-2        # expected/observed ratio agrees (cov-only ~1e-3, combined ~5e-3)
ATOL_AUC = 1.5e-2     # known ~5e-3 systematic R-vs-py difference in the IPW-AUC estimator
# The Hosmer-Lemeshow chi-square magnitude differs by ~15% because R and py use
# different weighted risk-score binning; we instead require the calibration
# *conclusion* (significant miscalibration at this level) to agree.
HL_ALPHA = 0.05


def load_golden(name):
    """Load a golden reference JSON file produced by the R generators."""
    with open(EXPECTED_DIR / name) as handle:
        return json.load(handle)


def read_profile(result):
    """Reconstruct the per-subject DataFrame from a py-icare result dict.

    py-icare returns the profile as a records-oriented JSON *string*; pandas 3.0
    requires it be wrapped in a file-like object.
    """
    return pd.read_json(io.StringIO(result["profile"]), orient="records")


def reference_population_risks(result, interval_index=0):
    """Pull the reference-population risk list out of a py-icare result dict."""
    return np.asarray(
        result["reference_risks"][interval_index]["population_risks"], dtype=float
    )


def summarize_distribution(values):
    """Order-independent summary matching the R ``summarize_dist`` helper.

    Uses numpy's default linear quantile method, which equals R's default
    (type 7), and the sample standard deviation (``ddof=1``) to match R's
    ``sd()``.
    """
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "q1": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "q3": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
        "sd": float(np.std(x, ddof=1)),
    }


def assert_distribution_close(py_values, golden_summary, atol, keys=None):
    """Assert a py distribution's summary matches the golden summary."""
    if keys is None:
        keys = ("min", "q1", "median", "mean", "q3", "max")
    py_summary = summarize_distribution(py_values)
    diffs = {
        key: (py_summary[key], golden_summary[key], abs(py_summary[key] - golden_summary[key]))
        for key in keys
    }
    failures = {key: vals for key, vals in diffs.items() if vals[2] > atol}
    assert not failures, f"distribution summary mismatch (atol={atol}): {failures}"
