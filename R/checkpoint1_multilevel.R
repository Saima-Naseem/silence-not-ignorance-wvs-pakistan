# =============================================================================
# Silence, not ignorance?  R CHECKPOINT 1: multilevel models for H2 and H4
# -----------------------------------------------------------------------------
# Run AFTER the Python Phase A scripts (01_prepare_data.py, 02_descriptives.py).
# Needs: data/analysis_wide.csv and data/analysis_long.csv
#
# How to run:
#   1. Open RStudio and set the working directory to the project folder
#      (the folder that contains data/, python/, R/, results/):
#        setwd("C:/path/to/wvs_dk_study")
#   2. First time only, install packages:
#        install.packages(c("lme4", "marginaleffects", "dplyr", "readr"))
#   3. Run:  source("R/checkpoint1_multilevel.R")
#   4. Send back the file results/R_checkpoint1_output.txt (or paste its text).
#
# Expected run time: about 5-20 minutes, depending on your computer.
# =============================================================================

suppressPackageStartupMessages({
  library(lme4); library(marginaleffects); library(dplyr); library(readr)
})
set.seed(2018)
dir.create("results", showWarnings = FALSE)
sink("results/R_checkpoint1_output.txt", split = TRUE)
cat("R CHECKPOINT 1 output\n", format(Sys.time()), "\n", R.version.string, "\n")
cat("lme4", as.character(packageVersion("lme4")),
    "| marginaleffects", as.character(packageVersion("marginaleffects")), "\n\n")

# ---- Data -------------------------------------------------------------------
wide <- read_csv("data/analysis_wide.csv", show_col_types = FALSE)
long <- read_csv("data/analysis_long.csv", show_col_types = FALSE)

z <- function(x) as.numeric(scale(x))
ctrl <- wide %>%
  transmute(id, age_z = z(age), educ_z = z(education_isced), literate,
            income_z = z(income_scale), employed, urban,
            province = factor(province, levels = c("Punjab", "Sindh", "KP", "Balochistan")),
            attention_z = z(attention_index), digital_z = z(news_digital), tv_z = z(news_tv))

dat <- long %>%
  inner_join(ctrl, by = "id") %>%
  mutate(id = factor(id), item = factor(item), psu = factor(psu))
cc <- complete.cases(dat[, c("age_z", "educ_z", "income_z", "employed", "attention_z", "digital_z", "tv_z")])
dat <- dat[cc, ]
cat("Respondents in complete-case sample:", n_distinct(dat$id), "\n")
cat("Rows (respondent x item):", nrow(dat), "\n\n")

ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# ---- H2: placebo -- domestic political vs non-political items ----------------
h2 <- dat %>% filter(item_set %in% c("domestic_political", "non_political"))
cat("==== H2 sample: domestic political (59) + non-political (62) items ====\n")
cat("Rows:", nrow(h2), "\n\n")

# H2-M1: female x political item, with respondent-level controls
m_h2a <- glmer(dk ~ female * political_item + age_z + educ_z + literate + income_z +
                 employed + urban + province + (1 | id) + (1 | psu) + (1 | item),
               data = h2, family = binomial, control = ctl, nAGQ = 0)
cat("---- H2-M1: female x political_item (controls as main effects) ----\n")
print(summary(m_h2a)$coefficients, digits = 4)
cat("\nOdds ratios (fixed effects):\n")
print(round(exp(fixef(m_h2a)), 3))

# H2-M2: also let education, literacy and awareness affect political items differently.
# If female x political_item stays strong here, the gap is not just a
# resources/awareness difference -> the silence test inside the placebo design.
m_h2b <- glmer(dk ~ female * political_item +
                 political_item * (educ_z + literate + attention_z + digital_z + tv_z) +
                 age_z + income_z + employed + urban + province +
                 (1 | id) + (1 | psu) + (1 | item),
               data = h2, family = binomial, control = ctl, nAGQ = 0)
cat("\n---- H2-M2: + political_item x (education, literacy, attention, digital news, TV news) ----\n")
print(summary(m_h2b)$coefficients, digits = 4)
cat("\nOdds ratios (fixed effects):\n")
print(round(exp(fixef(m_h2b)), 3))

cat("\nPredicted probability of 'don't know' by sex and item type (H2-M2, average over sample):\n")
print(avg_predictions(m_h2b, by = c("political_item", "female"), re.form = NA))
cat("\nGender gap (women - men) by item type, percentage-point scale (H2-M2):\n")
print(avg_comparisons(m_h2b, variables = "female", by = "political_item", re.form = NA))

# ---- H4: topic gradient within domestic political items ---------------------
h4 <- dat %>% filter(item_set == "domestic_political") %>%
  mutate(item_block = factor(item_block, levels = c("democracy", "regime_preferences",
         "confidence_institutions", "corruption", "surveillance", "electoral_integrity")))
cat("\n\n==== H4 sample: domestic political items only ====\n")
cat("Rows:", nrow(h4), "\n\n")

m_h4_0 <- glmer(dk ~ female + item_block + age_z + educ_z + literate + income_z + employed +
                  urban + province + attention_z + digital_z + tv_z +
                  (1 | id) + (1 | psu) + (1 | item),
                data = h4, family = binomial, control = ctl, nAGQ = 0)
m_h4_1 <- update(m_h4_0, . ~ . + female:item_block)
cat("---- H4: joint test of female x item_block (likelihood-ratio test) ----\n")
print(anova(m_h4_0, m_h4_1))
cat("\nCoefficients, H4 model with interaction:\n")
print(summary(m_h4_1)$coefficients, digits = 4)
cat("\nGender gap (women - men) in probability of 'don't know', by block:\n")
print(avg_comparisons(m_h4_1, variables = "female", by = "item_block", re.form = NA))
cat("\nGender gap as odds ratio, by block:\n")
print(avg_comparisons(m_h4_1, variables = "female", by = "item_block", re.form = NA,
                      comparison = "lnoravg", transform = exp))

# ---- Convergence notes ------------------------------------------------------
cat("\n\n==== Convergence messages ====\n")
for (nm in c("m_h2a", "m_h2b", "m_h4_0", "m_h4_1")) {
  msg <- get(nm)@optinfo$conv$lme4$messages
  cat(nm, ":", if (is.null(msg)) "OK" else paste(msg, collapse = "; "), "\n")
}
cat("\nNote: models use nAGQ = 0 for speed. For the final paper, re-run the key model\n",
    "with nAGQ = 1 (slower) and confirm the estimates are essentially unchanged.\n")
sink()
cat("\nDone. Output saved to results/R_checkpoint1_output.txt\n")
