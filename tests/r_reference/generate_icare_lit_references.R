# Generate R iCARE golden reference outputs for the iCARE-Lit model.
#
# Unlike BPC3, iCARE-Lit is NOT shipped with the R package, so we translate the
# py-icare (Patsy) model specification in `data/iCARE-Lit/` into R-native form
# using tests/r_reference/helpers.R. The translation is verified: build_icare_lit_model
# asserts every log odds ratio maps to exactly one design-matrix column, and the
# resulting risks/linear-predictors match py-icare to ~6 significant figures.
#
# iCARE-Lit is an age-stratified model with two sub-models: lt50 (age < 50) and
# ge50 (age >= 50, includes an HRT x BMI interaction). The split-interval workflow
# combines them at the age-50 cutpoint.
#
# Run from the repository root:
#     Rscript tests/r_reference/generate_icare_lit_references.R

source("tests/r_reference/r_utils.R")
source("tests/r_reference/helpers.R")

L <- "data/iCARE-Lit/"
cat("Generating iCARE-Lit golden references...\n")

inc <- read_rate_matrix(file.path(L, "age_specific_breast_cancer_incidence_rates.csv"))
comp <- read_rate_matrix(file.path(L, "age_specific_all_cause_mortality_rates.csv"))

# Query fixtures: the first N rows of each reference population, with an id
# column, committed so the Python suite scores the identical individuals.
N_QUERY <- 50
write_query_fixture <- function(ref_path, out_path) {
  ref <- read.csv(file.path(L, ref_path), stringsAsFactors = FALSE, check.names = FALSE)
  q <- head(ref, N_QUERY)
  q <- cbind(id = sprintf("Q%02d", seq_len(nrow(q)) - 1L), q)
  write.csv(q, out_path, row.names = FALSE)
  cat("  wrote", out_path, "\n")
}
Q_LT50 <- file.path(FIXTURES_DIR, "icare_lit_query_lt50.csv")
Q_GE50 <- file.path(FIXTURES_DIR, "icare_lit_query_ge50.csv")
write_query_fixture("reference_covariate_data_lt50.csv", Q_LT50)
write_query_fixture("reference_covariate_data_ge50.csv", Q_GE50)

# ---------------------------------------------------------------------------
# 1 & 2. Covariate-only risk for each sub-model (deterministic anchors).
# ---------------------------------------------------------------------------
gen_covariate_only <- function(tag, formula_file, log_or_file, ref_file, query_path,
                               age_start, age_length) {
  model <- build_icare_lit_model(file.path(L, formula_file), file.path(L, log_or_file))
  cat("  ", tag, "design columns:", nrow(model$log.RR), "\n")
  reference <- read_icare_lit_data(file.path(L, ref_file), model$factor_levels)
  query <- read_icare_lit_data(query_path, model$factor_levels)
  query$id <- NULL
  res <- computeAbsoluteRisk(
    model.formula = model$formula, model.cov.info = model$cov.info, model.log.RR = model$log.RR,
    model.ref.dataset = reference, model.disease.incidence.rates = inc,
    model.competing.incidence.rates = comp,
    apply.age.start = age_start, apply.age.interval.length = age_length,
    apply.cov.profile = query, return.lp = TRUE, return.refs.risk = TRUE
  )
  write_golden(list(
    age_start = age_start, age_interval_length = age_length,
    risks = as.numeric(res$risk), linear_predictors = as.numeric(res$lps),
    reference_risk_summary = summarize_dist(res$refs.risk)
  ), paste0("icare_lit_covariate_only_", tag, ".json"))
}
gen_covariate_only("lt50", "model_formula_lt50.txt", "model_log_odds_ratios_lt50.json",
                   "reference_covariate_data_lt50.csv", Q_LT50, 40, 10)
gen_covariate_only("ge50", "model_formula_ge50.txt", "model_log_odds_ratios_ge50.json",
                   "reference_covariate_data_ge50.csv", Q_GE50, 50, 20)

# ---------------------------------------------------------------------------
# 3. Split interval: lt50 model before age 50, ge50 model after (deterministic).
#    The before/after sub-models use different covariate sets, so the query
#    pairs the lt50-format and ge50-format profiles row-wise.
# ---------------------------------------------------------------------------
m_lt <- build_icare_lit_model(file.path(L, "model_formula_lt50.txt"),
                              file.path(L, "model_log_odds_ratios_lt50.json"))
m_ge <- build_icare_lit_model(file.path(L, "model_formula_ge50.txt"),
                              file.path(L, "model_log_odds_ratios_ge50.json"))
ref_lt <- read_icare_lit_data(file.path(L, "reference_covariate_data_lt50.csv"), m_lt$factor_levels)
ref_ge <- read_icare_lit_data(file.path(L, "reference_covariate_data_ge50.csv"), m_ge$factor_levels)
q_lt <- read_icare_lit_data(Q_LT50, m_lt$factor_levels); q_lt$id <- NULL
q_ge <- read_icare_lit_data(Q_GE50, m_ge$factor_levels); q_ge$id <- NULL

res_split <- computeAbsoluteRiskSplitInterval(
  apply.age.start = 40, apply.age.interval.length = 20, cut.time = 50,
  model.formula = m_lt$formula, model.cov.info = m_lt$cov.info, model.log.RR = m_lt$log.RR,
  model.ref.dataset = ref_lt, apply.cov.profile = q_lt,
  model.formula.2 = m_ge$formula, model.cov.info.2 = m_ge$cov.info, model.log.RR.2 = m_ge$log.RR,
  model.ref.dataset.2 = ref_ge, apply.cov.profile.2 = q_ge,
  model.disease.incidence.rates = inc, model.competing.incidence.rates = comp
)
write_golden(list(
  age_start = 40, age_interval_length = 20, cutpoint = 50,
  risks = as.numeric(res_split$risk)
), "icare_lit_split_interval.json")

# ---------------------------------------------------------------------------
# 4. Model validation: full cohort (no sampling weights) with the ge50 model.
#    The cohort is subsampled to N_VAL for tractable run time; the same subset is
#    exported as fixtures so the Python suite validates the identical individuals.
# ---------------------------------------------------------------------------
N_VAL <- 5000
cov_all <- read.csv(file.path(L, "validation_cohort_covariate_data.csv"),
                    stringsAsFactors = FALSE, check.names = FALSE)
out_all <- read.csv(file.path(L, "validation_cohort_data.csv"),
                    stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(identical(cov_all$id[seq_len(N_VAL)], out_all$id[seq_len(N_VAL)]))
cov_sub <- head(cov_all, N_VAL)
out_sub <- head(out_all, N_VAL)

# Export the subsampled cohort for Python (keeps data/ untouched).
write.csv(out_sub, file.path(FIXTURES_DIR, "icare_lit_validation_study.csv"), row.names = FALSE)
write.csv(cov_sub, file.path(FIXTURES_DIR, "icare_lit_validation_covariates.csv"), row.names = FALSE)
cat("  wrote iCARE-Lit validation fixtures (n =", N_VAL, ")\n")

study <- data.frame(
  observed.outcome = out_sub$observed_outcome,
  study.entry.age = out_sub$study_entry_age,
  study.exit.age = out_sub$study_exit_age,
  observed.followup = out_sub$study_exit_age - out_sub$study_entry_age,
  time.of.onset = ifelse(tolower(out_sub$time_of_onset) == "inf", Inf,
                         suppressWarnings(as.numeric(out_sub$time_of_onset)))
)
prof_ge <- cov_sub
for (v in names(m_ge$factor_levels)) prof_ge[[v]] <- factor(prof_ge[[v]], levels = m_ge$factor_levels[[v]])
prof_ge$id <- NULL

risk_model <- list(
  model.formula = m_ge$formula, model.cov.info = m_ge$cov.info, model.log.RR = m_ge$log.RR,
  model.ref.dataset = ref_ge, model.disease.incidence.rates = inc,
  model.competing.incidence.rates = comp,
  apply.cov.profile = prof_ge, n.imp = 5, use.c.code = 1, return.lp = TRUE, return.refs.risk = TRUE
)
set.seed(GOLDEN_SEED)
out_val <- ModelValidation(
  study.data = study, total.followup.validation = TRUE, predicted.risk.interval = NULL,
  iCARE.model.object = risk_model, number.of.percentiles = 10
)
# Recompute the AUC with the 0.5-tie convention (see weighted_auc_with_ties); this is
# a full cohort (no sampling weights), so the estimator is unweighted.
write_golden(
  apply_tie_aware_auc(validation_metrics(out_val), out_val, sampling.weights = NULL),
  "icare_lit_validation.json"
)

cat("iCARE-Lit golden references complete.\n")
