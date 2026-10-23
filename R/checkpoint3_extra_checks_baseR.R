# =============================================================================
# Silence, not ignorance?  R CHECKPOINT 3 (optional): verification of the extra checks
# -----------------------------------------------------------------------------
# Re-estimates the main models of python/11_extra_checks.py with MASS::glm.nb
# and prints R and Python incidence rate ratios side by side.
# Needs: data/analysis_wide.csv, data/extra_vars.csv, results/11_extra_checks.csv
# Run:   source("R/checkpoint3_extra_checks_baseR.R")
# Output: results/R_checkpoint3_output.txt
# =============================================================================
options(width = 120)
suppressPackageStartupMessages(library(MASS))
sink("results/R_checkpoint3_output.txt", split = TRUE)
cat("R CHECKPOINT 3 output\n", format(Sys.time()), "\n", R.version.string, "\n\n")

w  <- merge(read.csv("data/analysis_wide.csv"), read.csv("data/extra_vars.csv"), by = "id", sort = FALSE)
py <- read.csv("results/11_extra_checks.csv")
z  <- function(x) as.numeric(scale(x))
q222 <- ifelse(w$Q222 > 0, w$Q222, NA)
trad <- function(x) ifelse(x > 0, 5 - x, NA)
d <- data.frame(psu = w$psu, female = w$female, y = w$dk_count_domestic_political, y_ei = w$dk_ei,
  y_non = w$dk_count_non_political, age_z = z(w$age), educ_z = z(w$education_isced), literate = w$literate,
  income_z = z(w$income_scale), employed = w$employed, urban = w$urban,
  province = factor(w$province, levels = c("Punjab", "Sindh", "KP", "Balochistan")),
  attention_z = z(w$attention_index), digital_z = z(w$news_digital), tv_z = z(w$news_tv),
  regular_voter = as.numeric(q222 %in% c(1, 2)), never_votes = as.numeric(q222 == 3), not_allowed = as.numeric(q222 == 4),
  vote_ok = !is.na(q222), Q29_z = z(trad(w$Q29)))
d$never_votes[!d$vote_ok] <- NA; d$not_allowed[!d$vote_ok] <- NA; d$regular_voter[!d$vote_ok] <- NA
CORE <- c("y", "y_ei", "female", "age_z", "educ_z", "literate", "income_z", "employed", "urban", "province",
          "attention_z", "digital_z", "tv_z")
CTRL <- "age_z + educ_z + literate + income_z + employed + urban + province + attention_z + digital_z + tv_z"

cl_vcov_nb <- function(m, cluster) {
  X <- model.matrix(m); mu <- fitted(m); th <- m$theta
  U <- rowsum(X * (m$y - mu) / (1 + mu / th), cluster)
  B <- summary(m, dispersion = 1)$cov.unscaled; G <- nrow(U)
  B %*% crossprod(U) %*% B * G / (G - 1)
}
show <- function(label, m, term, data, py_model, py_term) {
  V <- cl_vcov_nb(m, data$psu); b <- coef(m)[term]; se <- sqrt(V[term, term])
  pyv <- py$estimate[py$model == py_model & py$term == py_term]
  cat(sprintf("%-58s R IRR %6.3f | Python IRR %6.3f | R p = %.4f | n = %d\n", label, exp(b), pyv, 2 * pnorm(-abs(b / se)), nrow(data)))
}
d1 <- d[complete.cases(d[, c(CORE, "regular_voter")]), ]
m <- glm.nb(as.formula(paste("y_ei ~ female +", CTRL)), data = d1)
show("1a electoral integrity, base: female", m, "female", d1, "Electoral-integrity DK: base model", "female")
m <- glm.nb(as.formula(paste("y_ei ~ female +", CTRL, "+ never_votes + not_allowed")), data = d1)
show("1a electoral integrity, + voting: female", m, "female", d1, "Electoral-integrity DK: + voting status", "female")
show("1a electoral integrity, + voting: not allowed", m, "not_allowed", d1, "Electoral-integrity DK: + voting status", "not_allowed")
v <- d1[d1$regular_voter == 1, ]
m <- glm.nb(as.formula(paste("y_ei ~ female +", CTRL)), data = v)
show("1b regular voters, electoral integrity: female", m, "female", v, "Regular voters only: electoral-integrity DK", "female")
d2 <- d[complete.cases(d[, c(CORE, "Q29_z")]), ]
wo <- d2[d2$female == 1, ]; mo <- d2[d2$female == 0, ]
m <- glm.nb(as.formula(paste("y ~ Q29_z +", CTRL)), data = wo)
show("2a women, domestic DK: Q29", m, "Q29_z", wo, "Women only: domestic DK on Q29 (traditional, z)", "Q29_z")
m <- glm.nb(as.formula(paste("y_non ~ Q29_z +", CTRL)), data = wo)
show("2b women, NON-political DK: Q29 (placebo)", m, "Q29_z", wo, "Women only: NON-political DK on Q29 (placebo)", "Q29_z")
m <- glm.nb(as.formula(paste("y ~ Q29_z +", CTRL)), data = mo)
show("2a men, domestic DK: Q29", m, "Q29_z", mo, "Men only: domestic DK on Q29 (comparison)", "Q29_z")
m <- glm.nb(as.formula(paste("y ~ female * Q29_z +", CTRL)), data = d2)
show("2c full sample: female x Q29", m, "female:Q29_z", d2, "Full sample: female x Q29 on domestic DK", "female:Q29_z")
cat("\nIRRs should match to about 3 decimals; p-values may differ slightly (see checkpoint 2 notes).\n")
sink()
cat("\nDone. Output saved to results/R_checkpoint3_output.txt\n")
