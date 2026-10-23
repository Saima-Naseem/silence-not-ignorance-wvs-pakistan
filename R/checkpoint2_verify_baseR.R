# =============================================================================
# Silence, not ignorance?  R CHECKPOINT 2: independent verification of Phase B
# -----------------------------------------------------------------------------
# Re-estimates the key Phase B negative binomial models with MASS::glm.nb
# (built into R) and prints them NEXT TO the Python results, so you can check
# that two independent implementations agree.
#
# Needs: data/analysis_wide.csv, data/town.csv, and the Python results files
#        results/05_stepwise_female.csv, results/05_negbin_female_x_attention.csv,
#        results/06_negbin_female_x_privacy.csv, results/07_robustness.csv
# Run:   source("R/checkpoint2_verify_baseR.R")
# Output: results/R_checkpoint2_output.txt
#
# What to expect: coefficients should match to about 3 decimals. Robust SEs can
# differ slightly in the 3rd decimal, because glm.nb treats the dispersion
# parameter as fixed when computing the sandwich, while Python estimates it jointly.
# =============================================================================

options(width = 120)
suppressPackageStartupMessages(library(MASS))
dir.create("results", showWarnings = FALSE)
sink("results/R_checkpoint2_output.txt", split = TRUE)
cat("R CHECKPOINT 2 output\n", format(Sys.time()), "\n", R.version.string, "\n\n")

w    <- read.csv("data/analysis_wide.csv")
town <- read.csv("data/town.csv")
w    <- merge(w, town, by = "id", sort = FALSE)
z <- function(x) as.numeric(scale(x))
PROV <- c("Punjab", "Sindh", "KP", "Balochistan")

d_all <- data.frame(
  psu = w$psu, female = w$female, y = w$dk_count_domestic_political,
  age_z = z(w$age), educ_z = z(w$education_isced), literate = w$literate,
  income_z = z(w$income_scale), employed = w$employed, urban = w$urban,
  province = factor(w$province, levels = PROV), town = factor(w$town),
  attention_z = z(w$attention_index), digital_z = z(w$news_digital), tv_z = z(w$news_tv),
  others_present = w$others_present, resp_interest_low = w$resp_interest_low,
  duration_z = z(w$duration_min))
d <- d_all[complete.cases(d_all), ]
cat("Analysis sample (should be 1845):", nrow(d), "\n\n")

# Cluster-robust covariance for glm.nb (sampling unit clusters)
cl_vcov_nb <- function(m, cluster) {
  X <- model.matrix(m); mu <- fitted(m); th <- m$theta
  U <- rowsum(X * (m$y - mu) / (1 + mu / th), cluster)
  B <- summary(m, dispersion = 1)$cov.unscaled; G <- nrow(U)
  B %*% crossprod(U) %*% B * G / (G - 1)
}
row_for <- function(m, term, cluster) {
  V <- cl_vcov_nb(m, cluster); b <- coef(m)[term]; se <- sqrt(V[term, term])
  c(estimate = unname(b), robust_se = unname(se), IRR = unname(exp(b)), p = unname(2 * pnorm(-abs(b / se))))
}
compare <- function(label, r_vals, py_est, py_se) {
  cat(sprintf("%-46s R est %8.4f | Py est %8.4f | R se %7.4f | Py se %7.4f | diff est %7.4f\n",
              label, r_vals["estimate"], py_est, r_vals["robust_se"], py_se, r_vals["estimate"] - py_est))
}

RES <- "age_z + educ_z + literate + income_z + employed + urban + province"
AWR <- "attention_z + digital_z + tv_z"
INT <- "others_present + resp_interest_low + duration_z"
forms <- list(
  M0 = y ~ female,
  M1 = as.formula(paste("y ~ female +", RES)),
  M2 = as.formula(paste("y ~ female +", RES, "+", AWR)),
  M3 = as.formula(paste("y ~ female +", RES, "+", AWR, "+", INT)))

cat("==== H1 and H3: nested negative binomial models, female coefficient ====\n")
py <- read.csv("results/05_stepwise_female.csv")
for (m in names(forms)) {
  fitm <- glm.nb(forms[[m]], data = d)
  r <- row_for(fitm, "female", d$psu)
  compare(paste(m, "female"), r, py$female_b[py$model == m], py$female_se[py$model == m])
  if (m == "M3") { cat("\nFull M3 (R, robust SE):\n"); V <- cl_vcov_nb(fitm, d$psu)
    print(round(cbind(estimate = coef(fitm), robust_se = sqrt(diag(V)), IRR = exp(coef(fitm))), 4)); cat("\n") }
}

cat("\n==== H3: female x political attention ====\n")
py_i <- read.csv("results/05_negbin_female_x_attention.csv", row.names = 1)
mi <- glm.nb(as.formula(paste("y ~ female +", RES, "+", AWR, "+ female:attention_z")), data = d)
for (tm in c("female", "female:attention_z"))
  compare(tm, row_for(mi, tm, d$psu), py_i[tm, "estimate"], py_i[tm, "se"])

cat("\n==== H5: female x others present during interview ====\n")
py_p <- read.csv("results/06_negbin_female_x_privacy.csv", row.names = 1)
mp <- glm.nb(as.formula(paste("y ~ female + others_present +", RES, "+", AWR,
                              "+ resp_interest_low + duration_z + female:others_present")), data = d)
for (tm in c("female", "others_present", "female:others_present"))
  compare(tm, row_for(mp, tm, d$psu), py_p[tm, "estimate"], py_p[tm, "se"])

cat("\n==== Robustness: town fixed effects (women vs men in the same town) ====\n")
d1 <- d_all[complete.cases(d_all[, c("y", "female", "age_z", "educ_z", "literate", "income_z",
                                      "employed", "urban", "province", "town", "psu")]), ]
both  <- tapply(d1$female, d1$town, function(s) length(unique(s)) > 1)
pos   <- tapply(d1$y, d1$town, sum) > 0
keep_towns <- names(both)[both & pos]
dt <- droplevels(d1[d1$town %in% keep_towns, ])
cat("Towns:", length(keep_towns), " respondents:", nrow(dt), " (Python: 24 towns, 1163 respondents)\n")
mt <- glm.nb(y ~ female + age_z + educ_z + literate + income_z + employed + urban + town, data = dt)
rt <- row_for(mt, "female", dt$psu)
rob <- read.csv("results/07_robustness.csv")
py_irr <- rob$H1_IRR_female[grepl("^town fixed effects", rob$variant)]
cat(sprintf("Town FE female IRR: R %.3f | Python %.3f | R robust 95%% CI %.2f-%.2f | p = %.4g\n",
            rt["IRR"], py_irr, exp(rt["estimate"] - 1.96 * rt["robust_se"]),
            exp(rt["estimate"] + 1.96 * rt["robust_se"]), rt["p"]))

cat("\nTheta (R) = 1/alpha (Python). Python alpha for M3 was about 1.905, i.e. theta about 0.525.\n")
sink()
cat("\nDone. Output saved to results/R_checkpoint2_output.txt\n")
