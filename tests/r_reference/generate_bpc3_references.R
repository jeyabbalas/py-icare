# Generate R iCARE golden reference outputs for the BPC3 breast-cancer model.
#
# The BPC3 model shipped in `data/BPC3/` is the SAME model as the iCARE package's
# built-in `bc_data` (verified: identical SNP odds ratios / names / incidence
# rates; covariate-only risks and linear predictors match to ~6 significant
# figures), just re-encoded (R uses integer decile codes, Python uses label
# strings). We therefore generate the R side directly from native `bc_data`,
# exactly as the package vignette does, and the Python suite runs the equivalent
# files in `data/BPC3/`.
#
# Run from the repository root:
#     Rscript tests/r_reference/generate_bpc3_references.R

source("tests/r_reference/r_utils.R")
data("bc_data", package = "iCARE")

cat("Generating BPC3 golden references...\n")

AGE_START <- 50
AGE_LENGTH <- 30

# ---------------------------------------------------------------------------
# 1. Covariate-only model (deterministic anchor: no SNPs, full profiles)
# ---------------------------------------------------------------------------
res_cov <- computeAbsoluteRisk(
  model.formula = bc_model_formula,
  model.cov.info = bc_model_cov_info,
  model.log.RR = bc_model_log_or,
  model.ref.dataset = ref_cov_dat,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  apply.age.start = AGE_START,
  apply.age.interval.length = AGE_LENGTH,
  apply.cov.profile = new_cov_prof,
  return.lp = TRUE,
  return.refs.risk = TRUE
)
write_golden(list(
  age_start = AGE_START,
  age_interval_length = AGE_LENGTH,
  risks = as.numeric(res_cov$risk),
  linear_predictors = as.numeric(res_cov$lps),
  reference_risk_summary = summarize_dist(res_cov$refs.risk)
), "bpc3_covariate_only.json")

# ---------------------------------------------------------------------------
# 2. Special SNP-only model, NO profile -> imputed reference distribution
#    (stochastic: compare distribution summaries only)
# ---------------------------------------------------------------------------
set.seed(GOLDEN_SEED)
res_snp_noprof <- computeAbsoluteRisk(
  model.snp.info = bc_72_snps,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  apply.age.start = AGE_START,
  apply.age.interval.length = AGE_LENGTH,
  return.refs.risk = TRUE
)
write_golden(list(
  age_start = AGE_START,
  age_interval_length = AGE_LENGTH,
  risk_summary = summarize_dist(res_snp_noprof$risk),
  reference_risk_summary = summarize_dist(res_snp_noprof$refs.risk)
), "bpc3_snp_only_no_profile.json")

# ---------------------------------------------------------------------------
# 3. SNP-only model WITH observed genotypes (stochastic: SNP imputation)
# ---------------------------------------------------------------------------
set.seed(GOLDEN_SEED)
res_snp_prof <- computeAbsoluteRisk(
  model.snp.info = bc_72_snps,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  apply.age.start = AGE_START,
  apply.age.interval.length = AGE_LENGTH,
  apply.snp.profile = new_snp_prof,
  return.refs.risk = TRUE
)
write_golden(list(
  age_start = AGE_START,
  age_interval_length = AGE_LENGTH,
  risks = as.numeric(res_snp_prof$risk),
  reference_risk_summary = summarize_dist(res_snp_prof$refs.risk)
), "bpc3_snp_only_with_profile.json")

# ---------------------------------------------------------------------------
# 4. Combined covariate + SNP model (stochastic: SNP imputation)
# ---------------------------------------------------------------------------
set.seed(GOLDEN_SEED)
res_comb <- computeAbsoluteRisk(
  model.formula = bc_model_formula,
  model.cov.info = bc_model_cov_info,
  model.snp.info = bc_72_snps,
  model.log.RR = bc_model_log_or,
  model.ref.dataset = ref_cov_dat,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  model.bin.fh.name = "famhist",
  apply.age.start = AGE_START,
  apply.age.interval.length = AGE_LENGTH,
  apply.cov.profile = new_cov_prof,
  apply.snp.profile = new_snp_prof,
  return.refs.risk = TRUE
)
write_golden(list(
  age_start = AGE_START,
  age_interval_length = AGE_LENGTH,
  risks = as.numeric(res_comb$risk),
  reference_risk_summary = summarize_dist(res_comb$refs.risk)
), "bpc3_combined.json")

# ---------------------------------------------------------------------------
# 5. Split interval: ages 30-70 split at 50, pre/post-50 betas & references
#    5a covariate-only (deterministic anchor); 5b combined with SNPs (stochastic)
# ---------------------------------------------------------------------------
SPLIT_START <- 30
SPLIT_LENGTH <- 40
CUT_TIME <- 50

res_split_cov <- computeAbsoluteRiskSplitInterval(
  apply.age.start = SPLIT_START,
  apply.age.interval.length = SPLIT_LENGTH,
  apply.cov.profile = new_cov_prof,
  model.formula = bc_model_formula,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  model.log.RR = bc_model_log_or,
  model.log.RR.2 = bc_model_log_or_post_50,
  model.ref.dataset = ref_cov_dat,
  model.ref.dataset.2 = ref_cov_dat_post_50,
  model.cov.info = bc_model_cov_info,
  cut.time = CUT_TIME
)
write_golden(list(
  age_start = SPLIT_START,
  age_interval_length = SPLIT_LENGTH,
  cutpoint = CUT_TIME,
  risks = as.numeric(res_split_cov$risk)
), "bpc3_split_interval_covariate_only.json")

set.seed(GOLDEN_SEED)
res_split_comb <- computeAbsoluteRiskSplitInterval(
  apply.age.start = SPLIT_START,
  apply.age.interval.length = SPLIT_LENGTH,
  apply.cov.profile = new_cov_prof,
  apply.snp.profile = new_snp_prof,
  model.formula = bc_model_formula,
  model.snp.info = bc_72_snps,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  model.log.RR = bc_model_log_or,
  model.log.RR.2 = bc_model_log_or_post_50,
  model.ref.dataset = ref_cov_dat,
  model.ref.dataset.2 = ref_cov_dat_post_50,
  model.cov.info = bc_model_cov_info,
  model.bin.fh.name = "famhist",
  cut.time = CUT_TIME
)
write_golden(list(
  age_start = SPLIT_START,
  age_interval_length = SPLIT_LENGTH,
  cutpoint = CUT_TIME,
  risks = as.numeric(res_split_comb$risk)
), "bpc3_split_interval_combined.json")

# ---------------------------------------------------------------------------
# 6. Model validation on the nested case-control study.
#    Sampling weights are recomputed via the vignette's inclusion (selection)
#    model -- the shipped `sampling.weights` column makes iCARE's own weighted
#    categorization fail, and the vignette overwrites it anyway. We export these
#    weights so the Python suite validates with the IDENTICAL weights.
# ---------------------------------------------------------------------------
vc <- validation.cohort.data
vc$inclusion <- 0
vc$inclusion[intersect(vc$id, validation.nested.case.control.data$id)] <- 1
vc$observed.followup <- vc$study.exit.age - vc$study.entry.age
selection.model <- glm(
  inclusion ~ observed.outcome * (study.entry.age + observed.followup),
  data = vc, family = binomial(link = "logit")
)
study <- validation.nested.case.control.data
study$sampling.weights <- selection.model$fitted.values[vc$inclusion == 1]

# Fixture so Python uses the exact same weights (keeps data/ untouched).
write.csv(
  data.frame(id = study$id, sampling_weights = study$sampling.weights),
  file.path(FIXTURES_DIR, "bpc3_nested_cc_glm_weights.csv"),
  row.names = FALSE
)
cat("  wrote", file.path(FIXTURES_DIR, "bpc3_nested_cc_glm_weights.csv"), "\n")

risk_model_cov <- list(
  model.formula = bc_model_formula,
  model.cov.info = bc_model_cov_info,
  model.log.RR = bc_model_log_or,
  model.ref.dataset = ref_cov_dat,
  model.disease.incidence.rates = bc_inc,
  model.competing.incidence.rates = mort_inc,
  model.bin.fh.name = "famhist",
  apply.cov.profile = study[, all.vars(bc_model_formula)[-1]],
  n.imp = 5, use.c.code = 1, return.lp = TRUE, return.refs.risk = TRUE
)
set.seed(GOLDEN_SEED)
out_val_cov <- ModelValidation(
  study.data = study, total.followup.validation = TRUE,
  predicted.risk.interval = NULL, iCARE.model.object = risk_model_cov,
  number.of.percentiles = 10
)
write_golden(validation_metrics(out_val_cov), "bpc3_validation_covariate_only.json")

risk_model_comb <- risk_model_cov
risk_model_comb$model.snp.info <- bc_72_snps
risk_model_comb$apply.snp.profile <- study[, bc_72_snps$snp.name]
set.seed(GOLDEN_SEED)
out_val_comb <- ModelValidation(
  study.data = study, total.followup.validation = TRUE,
  predicted.risk.interval = NULL, iCARE.model.object = risk_model_comb,
  number.of.percentiles = 10
)
write_golden(validation_metrics(out_val_comb), "bpc3_validation_combined.json")

cat("BPC3 golden references complete.\n")
