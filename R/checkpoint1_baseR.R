# =============================================================================
# Silence, not ignorance?  R CHECKPOINT 1 (base-R version, no lme4 needed)
# -----------------------------------------------------------------------------
# Uses only packages that ship with R: MASS and nlme.
# Needs: data/analysis_wide.csv and data/item_sets.csv (from Phase A)
#
# Run:  source("R/checkpoint1_baseR.R")
# Output: results/R_checkpoint1_output.txt
#
# Design: each respondent contributes one row per item type (H2) or per topic
# block (H4): the share of "don't know" answers, weighted by the number of items.
# Multilevel logistic model fitted by penalized quasi-likelihood (MASS::glmmPQL),
# random intercepts for sampling unit (psu) and respondent within psu.
# Cross-check: ordinary quasi-binomial GLM with cluster-robust standard errors by psu.
# =============================================================================

options(width = 120)
suppressPackageStartupMessages({ library(MASS); library(nlme) })
set.seed(2018)
dir.create("results", showWarnings = FALSE)
sink("results/R_checkpoint1_output.txt", split = TRUE)
cat("R CHECKPOINT 1 (base-R version) output\n", format(Sys.time()), "\n", R.version.string, "\n\n")

# ---- Data -------------------------------------------------------------------
wide  <- read.csv("data/analysis_wide.csv")
items <- read.csv("data/item_sets.csv")
z <- function(x) as.numeric(scale(x))

d <- data.frame(
  id = factor(wide$id), psu = factor(wide$psu), female = wide$female,
  age_z = z(wide$age), educ_z = z(wide$education_isced), literate = wide$literate,
  income_z = z(wide$income_scale), employed = wide$employed, urban = wide$urban,
  province = factor(wide$province, levels = c("Punjab", "Sindh", "KP", "Balochistan")),
  attention_z = z(wide$attention_index), digital_z = z(wide$news_digital), tv_z = z(wide$news_tv))
keep <- complete.cases(d)
d <- d[keep, ]; w <- wide[keep, ]
cat("Respondents in complete-case sample:", nrow(d),
    "(women:", sum(d$female), ", men:", sum(1 - d$female), ")\n\n")

ctrl <- "age_z + educ_z + literate + income_z + employed + urban + province"

# ---- Helper functions -------------------------------------------------------
# Average predicted gap (women - men) in probability of "don't know",
# with 95% interval from 2,000 simulated draws of the fixed effects.
gap_sim <- function(m, form, base, by_var = NULL, by_levels = NULL, ndraw = 2000) {
  tt <- delete.response(terms(form)); b <- fixef(m); V <- vcov(m)
  draws <- mvrnorm(ndraw, b, V)
  if (is.null(by_var)) by_levels <- "all"
  out <- data.frame()
  for (lv in by_levels) {
    nd <- base
    if (!is.null(by_var)) {
      nd[[by_var]] <- if (is.factor(base[[by_var]]))
        factor(rep(lv, nrow(nd)), levels = levels(base[[by_var]])) else lv
    }
    nd0 <- nd; nd0$female <- 0; nd1 <- nd; nd1$female <- 1
    X0 <- model.matrix(tt, nd0); X1 <- model.matrix(tt, nd1)
    stopifnot(identical(colnames(X0), names(b)))
    p0 <- mean(plogis(X0 %*% b)); p1 <- mean(plogis(X1 %*% b))
    g  <- colMeans(plogis(X1 %*% t(draws))) - colMeans(plogis(X0 %*% t(draws)))
    out <- rbind(out, data.frame(level = lv, men_pct = round(100 * p0, 2),
                 women_pct = round(100 * p1, 2), gap_pp = round(100 * (p1 - p0), 2),
                 ci_low = round(100 * quantile(g, .025), 2), ci_high = round(100 * quantile(g, .975), 2)))
  }
  rownames(out) <- NULL; out
}
# Joint Wald test for a group of coefficients
wald <- function(m, pattern) {
  b <- fixef(m); idx <- grep(pattern, names(b))
  W <- as.numeric(t(b[idx]) %*% solve(vcov(m)[idx, idx]) %*% b[idx])
  c(Wald_chi2 = round(W, 2), df = length(idx), p = signif(pchisq(W, length(idx), lower.tail = FALSE), 4))
}
# Cluster-robust covariance for a logit-link GLM
cl_vcov <- function(g, cluster) {
  X <- model.matrix(g); r <- g$prior.weights * (g$y - g$fitted.values)
  U <- rowsum(X * r, cluster); B <- summary(g)$cov.unscaled; G <- nrow(U)
  B %*% crossprod(U) %*% B * G / (G - 1)
}
or_table <- function(m) {
  tt <- summary(m)$tTable
  data.frame(estimate = round(tt[, "Value"], 4), se = round(tt[, "Std.Error"], 4),
             odds_ratio = round(exp(tt[, "Value"]), 3), p = signif(tt[, "p-value"], 4))
}

# ---- H2: placebo -- domestic political vs non-political ---------------------
h2 <- rbind(
  cbind(d, political = 1, share = w$dk_share_domestic_political, n_items = w$n_items_domestic_political),
  cbind(d, political = 0, share = w$dk_share_non_political,      n_items = w$n_items_non_political))
cat("==== H2: placebo design (2 rows per respondent) ====\n")

f_h2a <- as.formula(paste("share ~ female * political +", ctrl))
m_h2a <- glmmPQL(f_h2a, random = ~ 1 | psu/id, family = binomial,
                 data = h2, weights = n_items, verbose = FALSE)
cat("\n---- H2-M1: female x political item, controls as main effects ----\n")
print(or_table(m_h2a))

f_h2b <- as.formula(paste("share ~ female * political +",
  "political * (educ_z + literate + attention_z + digital_z + tv_z) +",
  "age_z + income_z + employed + urban + province"))
m_h2b <- glmmPQL(f_h2b, random = ~ 1 | psu/id, family = binomial,
                 data = h2, weights = n_items, verbose = FALSE)
cat("\n---- H2-M2: + political x (education, literacy, attention, digital news, TV news) ----\n")
print(or_table(m_h2b))

cat("\nPredicted 'don't know' % and gender gap by item type (H2-M2):\n")
print(gap_sim(m_h2b, f_h2b, d, by_var = "political", by_levels = c(0, 1)))

cat("\n---- Cross-check: quasi-binomial GLM, cluster-robust SE by psu (H2-M1) ----\n")
g_h2a <- glm(f_h2a, family = quasibinomial, data = h2, weights = n_items)
Vc <- cl_vcov(g_h2a, h2$psu); se <- sqrt(diag(Vc)); b <- coef(g_h2a)
print(data.frame(estimate = round(b, 4), robust_se = round(se, 4),
                 odds_ratio = round(exp(b), 3), p = signif(2 * pnorm(-abs(b / se)), 4)))

# ---- H4: topic gradient within domestic items -------------------------------
blk_levels <- c("democracy", "regime_preferences", "confidence_institutions",
                "corruption", "surveillance", "electoral_integrity")
blk_n <- table(items$item_block[items$item_set == "domestic_political"])
h4 <- do.call(rbind, lapply(blk_levels, function(b)
  cbind(d, block = b, share = w[[paste0("dk_share_block_", b)]], n_items = as.numeric(blk_n[b]))))
h4$block <- factor(h4$block, levels = blk_levels)
d$block  <- factor(blk_levels[1], levels = blk_levels)
cat("\n\n==== H4: topic gradient (6 rows per respondent) ====\n")
cat("Items per block:\n"); print(blk_n[blk_levels])

f_h4 <- as.formula(paste("share ~ female * block +", ctrl, "+ attention_z + digital_z + tv_z"))
m_h4 <- glmmPQL(f_h4, random = ~ 1 | psu/id, family = binomial,
                data = h4, weights = n_items, verbose = FALSE)
cat("\n---- H4 model coefficients ----\n")
print(or_table(m_h4))
cat("\nJoint Wald test: does the gender gap differ across blocks? (female x block terms)\n")
print(wald(m_h4, "^female:block"))
cat("\nPredicted 'don't know' % and gender gap by block:\n")
print(gap_sim(m_h4, f_h4, d, by_var = "block", by_levels = blk_levels))

cat("\nNotes: glmmPQL is an approximate (penalized quasi-likelihood) estimator; with ~60 items\n",
    "per respondent the approximation is good. p-values for fixed effects come from nlme t-tests.\n")
sink()
cat("\nDone. Output saved to results/R_checkpoint1_output.txt\n")
