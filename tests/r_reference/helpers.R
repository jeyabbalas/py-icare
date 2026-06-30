# Translate a py-icare (Patsy) model specification into the R-native form that
# the iCARE package expects, for the iCARE-Lit model (which, unlike BPC3, is not
# shipped with the R package).
#
# py-icare specifies a model as:
#   * a Patsy formula text file, e.g.  C(parity, levels=['0','1','2','>=3']) + height
#   * a JSON of log odds ratios keyed by Patsy design-matrix column names, e.g.
#       "C(parity, levels=['0','1','2','>=3'])[T.1]": -0.139
#
# R iCARE specifies the same model as:
#   * model.formula        e.g.  diagnosis ~ as.factor(parity) + height
#   * model.cov.info       a list giving each variable's type and factor levels
#   * model.log.RR         a one-column matrix whose rownames equal, in order,
#                          the columns of model.matrix(model.formula, data)
#
# The translation is mostly mechanical: C(var, levels=[...]) -> as.factor(var),
# with factor levels carried over in Patsy order (first level = reference). We
# obtain the exact design-column order from iCARE's own internal helper so that
# check_design_matrix() is satisfied, and assert that every beta maps to exactly
# one design column.

suppressMessages(library(iCARE))
suppressMessages(library(jsonlite))

# Read a Patsy formula text file (right-hand side only) into one whitespace-
# normalized line.
read_formula_text <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = " ")
  trimws(gsub("[[:space:]]+", " ", txt))
}

# Regex matching a Patsy categorical term  C(<var>, levels=[<...>])
.PATSY_C <- "C\\(\\s*([^,]+?)\\s*,\\s*levels=\\[([^\\]]*)\\]\\s*\\)"

# Extract ordered character levels for every C(var, levels=[...]) in the RHS.
extract_factor_levels <- function(patsy_rhs) {
  hits <- regmatches(patsy_rhs, gregexpr(.PATSY_C, patsy_rhs, perl = TRUE))[[1]]
  out <- list()
  for (hit in hits) {
    g <- regmatches(hit, regexec(.PATSY_C, hit, perl = TRUE))[[1]]
    var <- trimws(g[2])
    items <- trimws(strsplit(g[3], ",")[[1]])
    items <- gsub("^['\"]|['\"]$", "", items)
    out[[var]] <- items
  }
  out
}

# Convert a Patsy RHS to an R formula RHS: C(var, levels=[...]) -> as.factor(var).
# Operators (+, *, :) and grouping parentheses are identical in both grammars.
patsy_rhs_to_r <- function(patsy_rhs) {
  gsub(.PATSY_C, "as.factor(\\1)", patsy_rhs, perl = TRUE)
}

# Convert one Patsy design-column key to its R model.matrix column name, e.g.
#   C(parity, levels=[...])[T.1]                       -> as.factor(parity)1
#   hrt_type                                           -> hrt_type
#   C(hrt, levels=[...])[T.current]:C(bmi, ...)[T.>=30]-> as.factor(hrt)current:as.factor(bmi)>=30
patsy_key_to_r_name <- function(key) {
  term <- "^C\\(\\s*([^,]+?)\\s*,\\s*levels=\\[.*\\]\\)\\[T\\.(.+)\\]$"
  parts <- strsplit(key, ":", fixed = TRUE)[[1]]
  r_parts <- vapply(parts, function(p) {
    g <- regmatches(p, regexec(term, p, perl = TRUE))[[1]]
    if (length(g) == 3) paste0("as.factor(", trimws(g[2]), ")", g[3]) else p
  }, character(1))
  paste(r_parts, collapse = ":")
}

# Build the R iCARE model spec from a Patsy formula file and a log-OR JSON file.
build_icare_lit_model <- function(formula_path, log_or_path, response = "diagnosis") {
  patsy_rhs <- read_formula_text(formula_path)
  factor_levels <- extract_factor_levels(patsy_rhs)
  formula <- as.formula(paste(response, "~", patsy_rhs_to_r(patsy_rhs)))

  variables <- all.vars(formula)[-1]
  cov.info <- lapply(variables, function(v) {
    if (v %in% names(factor_levels)) {
      list(name = v, type = "factor", levels = factor_levels[[v]])
    } else {
      list(name = v, type = "continuous")
    }
  })

  # Exact design-column order iCARE will build (so check_design_matrix passes).
  design_names <- iCARE:::get_beta_given_names(cov.info, formula)

  betas <- fromJSON(log_or_path)
  beta_lookup <- setNames(
    as.numeric(unlist(betas)),
    vapply(names(betas), patsy_key_to_r_name, character(1))
  )

  missing <- setdiff(design_names, names(beta_lookup))
  if (length(missing) > 0) {
    stop(paste("Translation error: no beta for design column(s):",
               paste(missing, collapse = ", ")))
  }
  unused <- setdiff(names(beta_lookup), design_names)
  if (length(unused) > 0) {
    stop(paste("Translation error: beta(s) with no matching design column:",
               paste(unused, collapse = ", ")))
  }

  log_rr <- matrix(beta_lookup[design_names], ncol = 1,
                   dimnames = list(design_names, NULL))
  list(formula = formula, cov.info = cov.info, log.RR = log_rr,
       factor_levels = factor_levels)
}

# Read an iCARE-Lit covariate CSV, coercing factor columns to the model's level
# order so model.matrix() produces the column order the beta names assume.
read_icare_lit_data <- function(path, factor_levels) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  for (v in names(factor_levels)) {
    if (v %in% names(df)) df[[v]] <- factor(df[[v]], levels = factor_levels[[v]])
  }
  df
}

# Read an "age,rate" CSV as the two-column matrix iCARE expects.
read_rate_matrix <- function(path) as.matrix(read.csv(path))
