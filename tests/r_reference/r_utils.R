# Shared helpers for generating R iCARE golden reference files.
#
# These scripts produce the expected outputs that the Python (`py-icare`) test
# suite is compared against. Run them from the repository root, e.g.
#     Rscript tests/r_reference/generate_bpc3_references.R
#
# Requires: R with the Bioconductor `iCARE` package and `jsonlite` installed.

suppressMessages(library(iCARE))
suppressMessages(library(jsonlite))

# Output locations, relative to the repo root (scripts must be run from there).
EXPECTED_DIR <- "tests/r_reference/expected"
FIXTURES_DIR <- "tests/r_reference/fixtures"
dir.create(EXPECTED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIXTURES_DIR, recursive = TRUE, showWarnings = FALSE)

# Seed used for every stochastic call so the golden files are reproducible.
GOLDEN_SEED <- 50

# Order-independent, deterministic summary of a numeric vector. Used to compare
# whole risk distributions (e.g. the reference population) where the row order
# of the R and Python datasets is not guaranteed to match.
summarize_dist <- function(x) {
  x <- as.numeric(x)
  list(
    n = length(x),
    min = min(x),
    q1 = unname(quantile(x, 0.25)),
    median = median(x),
    mean = mean(x),
    q3 = unname(quantile(x, 0.75)),
    max = max(x),
    sd = sd(x)
  )
}

# Pull the headline metrics out of an iCARE `ModelValidation` result object.
validation_metrics <- function(out) {
  list(
    auc = as.numeric(out$AUC),
    auc_ci = as.numeric(out$CI_AUC),
    eo_ratio = as.numeric(out$Overall_Expected_to_Observed_Ratio),
    eo_ci = as.numeric(out$CI_Overall_Expected_to_Observed_Ratio),
    hl_chisq = as.numeric(out$Hosmer_Lemeshow_Results$statistic),
    hl_df = as.numeric(out$Hosmer_Lemeshow_Results$parameter),
    hl_pvalue = as.numeric(out$HL_pvalue),
    rr_chisq = as.numeric(out$RR_test_result$statistic),
    rr_df = as.numeric(out$RR_test_result$parameter),
    rr_pvalue = as.numeric(out$RR_test_pvalue)
  )
}

# Tie-aware AUC matching py-icare's `_calculate_auc` (icare/model_validation.py).
#
# iCARE's own AUC (and py-icare's, before this change) used a strict `>`, giving a
# tied case/control risk score 0 credit instead of the statistically correct 0.5
# (Mann-Whitney U / trapezoidal ROC). On the BPC3 nested case-control study R's
# risk score is coarse enough to produce many ties, so strict-`>` dragged R's AUC
# ~0.005 below py-icare's (essentially tie-free) value. We recompute the golden AUC
# from iCARE's OWN returned risk score / outcome with the 0.5-tie convention so the
# reference matches the fixed py-icare estimator; iCARE itself is left untouched.
#
# A (control, case) pair scores 1 if case > control, 0.5 if |case-control| <= tol,
# else 0. For a nested case-control study pass `sampling.weights` (freq = 1/weights,
# IPW via kronecker, exactly as iCARE and py-icare do); pass NULL for a full cohort.
# `tol` must match py-icare's AUC_TIE_TOLERANCE.
weighted_auc_with_ties <- function(lp, outcome, sampling.weights = NULL, tol = 1e-9) {
  lp <- as.numeric(lp)
  outcome <- as.integer(outcome)
  freq <- if (is.null(sampling.weights)) rep(1, length(lp)) else 1 / as.numeric(sampling.weights)
  stopifnot(length(lp) == length(outcome), length(freq) == length(lp))

  lp.cases <- lp[outcome == 1]
  lp.controls <- lp[outcome == 0]
  freq.cases <- freq[outcome == 1]
  freq.controls <- freq[outcome == 0]

  # indicator[control, case] in {0, 0.5, 1}; same (controls x cases) orientation as iCARE.
  gt <- vapply(lp.cases, function(x) as.numeric((x - lp.controls) > tol),
               numeric(length(lp.controls)))
  eq <- vapply(lp.cases, function(x) as.numeric(abs(x - lp.controls) <= tol),
               numeric(length(lp.controls)))
  indicator <- gt + 0.5 * eq

  weight.mat <- matrix(kronecker(freq.controls, freq.cases),
                       nrow = length(freq.controls), byrow = TRUE)
  sum(indicator * weight.mat) / sum(weight.mat)
}

# Overwrite an iCARE ModelValidation metrics list `m` with the tie-aware AUC, and
# re-center the (untested, cosmetic) Wald CI on the corrected estimate using
# iCARE's own AUC variance (which the tie credit changes only negligibly).
apply_tie_aware_auc <- function(m, out, sampling.weights = NULL) {
  m$auc <- weighted_auc_with_ties(
    out$Subject_Specific_Risk_Score,
    out$Subject_Specific_Observed_Outcome,
    sampling.weights
  )
  m$auc_ci <- m$auc + c(-1, 1) * 1.96 * sqrt(as.numeric(out$Variance_AUC))
  m
}

# Write an object to <EXPECTED_DIR>/<name> as pretty JSON with high precision.
write_golden <- function(obj, name) {
  path <- file.path(EXPECTED_DIR, name)
  write_json(obj, path, auto_unbox = TRUE, digits = 12, pretty = TRUE)
  cat("  wrote", path, "\n")
}
