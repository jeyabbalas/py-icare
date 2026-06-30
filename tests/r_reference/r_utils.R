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

# Write an object to <EXPECTED_DIR>/<name> as pretty JSON with high precision.
write_golden <- function(obj, name) {
  path <- file.path(EXPECTED_DIR, name)
  write_json(obj, path, auto_unbox = TRUE, digits = 12, pretty = TRUE)
  cat("  wrote", path, "\n")
}
