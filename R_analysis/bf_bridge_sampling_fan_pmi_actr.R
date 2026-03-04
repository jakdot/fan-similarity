library(tidyverse)
library(tidybayes)
library(here)
library(brms)
testing_all <- read_csv("../csv_files/online_testing_all.csv")

testing_all$charcount <- nchar(testing_all$test_sentence)
dotproduct <- read_csv("../csv_files/dotproducts_perword.csv")
actr <- read_csv("../csv_files/actr_activations_perword.csv")
print(dotproduct)
# Clean up data

# change rt from s into ms

testing_all$rt <- testing_all$rt * 1000

#testing_all <- testing_all %>% filter(rt > 0)

testing_all$fan <- ifelse(testing_all$fan_loc == 4 | testing_all$fan_pers == 4, 1, -1)
testing_all$type_fan <- NA
testing_all$type_fan <- ifelse(testing_all$fan == 0, 0, ifelse(testing_all$fan_pers == 4, 1, -1))
testing_all <- testing_all %>% group_by(test_sentence) %>% mutate(pers_word=as.character(strsplit(test_sentence, " ")[[1]][2]), loc_word=as.character(strsplit(test_sentence, " ")[[1]][6]))
testing_all$pp_num <- as.numeric(testing_all$pp_num)
testing_all$manipulated_fan <- ifelse(testing_all$pp_num %% 2 == 1, "location", "person" )
testing_all$pp_num <- as.character(testing_all$pp_num)
testing_all$fan_word <- ifelse(testing_all$manipulated_fan == "person", testing_all$pers_word, testing_all$loc_word)
testing_all$fan_foil <- ifelse(testing_all$condition == "target", 0, ifelse(testing_all$fan == 1, 1, -1))
testing_all$fan_target <- ifelse(testing_all$condition == "foil", 0, ifelse(testing_all$fan == 1, 1, -1))
testing_all$type_condition <- ifelse(testing_all$condition == "target", 1, -1)

print(testing_all, width=Inf)

# Exclude unreasonably long and short trials.

testing_all <- testing_all %>% filter(rt <= 90000 & rt > 200)

# Exclude people with accuracy lower than 60 percent.

testing_all <- testing_all %>% filter(rt > 0) # exclude rts that are 0
testing_all %>% group_by(pp_num) %>% summarize(accuracy = mean(correct)) %>% filter(accuracy <= 0.65)
testing_correct <-testing_all %>% filter(!(pp_num %in% c("43","65", "90"))) # excluding 43 and 65 because of low accuracy
testing_correct %>% group_by(pp_num) %>% summarize(accuracy = mean(correct))

testing_correct %>% group_by(fan_pers, fan_loc) %>% summarise(accuracy = mean(correct), se=sd(correct)/sqrt(length(correct)))

# Exclude trials slower than 8.3 seconds (3 sample sd away from the sample mean).

mean(testing_all$rt) + 3 * sd(testing_all$rt)

testing_faster <- testing_all %>% filter(rt <= mean(testing_all$rt) + 3 * sd(testing_all$rt)) # keeping only trials with a rt below 3 st.d. away from the mean
testing_faster_correct <- testing_correct %>% filter(rt <= mean(testing_all$rt) + 3 * sd(testing_all$rt)) # keeping only trials with a rt below 3 st.d. away from the mean

length(testing_faster_correct$rt)/length(testing_all$rt)
length(testing_correct$rt)/length(testing_all$rt)
length(testing_faster_correct$rt)/length(testing_correct$rt)

# Merge data

dotproduct$fan_word <- dotproduct$Word
dotproduct$pp_num <- dotproduct$List

dotproduct <- dotproduct %>% group_by(pp_num, fan_word) %>% summarise(pmi = mean(Dot_product) + log(5))

dotproduct$pp_num <- as.character(dotproduct$pp_num)

print(dotproduct)

merged_data <- inner_join(testing_faster_correct, dotproduct, by=c("fan_word", "pp_num"))

print(merged_data)

actr$fan_word <- actr$Word
actr$pp_num <- actr$List

actr <- actr %>% group_by(pp_num, fan_word) %>% summarise(actr = mean(ACT_R_activation))

actr$pp_num <- as.character(actr$pp_num)

print(actr)

merged_data <- inner_join(merged_data, actr, by=c("fan_word", "pp_num"))

print(merged_data)

# We subset by target (because these dot products were calculated on trained target-context pairs, 
# and only targets were trained).

merged_data$actr <- scale(merged_data$actr)[,1]

merged_data <- subset(merged_data, condition == "target")

write_csv(merged_data, file="../csv_files/online_testing_modeled_data.csv")

# Bayesian modeling

priors_full <- c(
  prior(normal(0, 15), class = Intercept),
  prior(normal(0, 2), class = b),
  prior(normal(0, 2), class = sigma),
  prior(normal(0, 2), class = sd),
  prior(lkj(2), class = cor)
)

priors <- c(
  prior(normal(7, 3), class = Intercept),
  prior(normal(0, 0.1), class = b),
  prior(normal(0, 2), class = sigma),
  prior(normal(0, 2), class = sd)
)

modeling_fan <- function(useddata, measure, adaptdelta=FALSE) {

  if (adaptdelta==FALSE) {
fit_model <- brm(
    paste0(measure, "~ 1 + fan + (1|pp_num) + (1|fan_word) "),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  save_all_pars = TRUE,
  prior = priors,
  iter = 10000,
  cores = 4
)
  } else {
fit_model <- brm(
    paste0(measure, "~ 1 + fan + (1|pp_num) + (1|fan_word)"),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  save_all_pars = TRUE,
  prior = priors,
  control = list(adapt_delta = 0.95),
  iter = 10000,
  cores = 4
)
  }

# output of the model

fit_model

}

modeling_pmi_full <- function(useddata, measure, adaptdelta=FALSE) {

  if (adaptdelta==FALSE) {
fit_model <- brm(
    paste0(measure, "~ 1 + pmi + (1 + pmi|pp_num) + (1+pmi|fan_word) "),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  prior = priors_full,
  iter = 3000,
  cores = 4
)
  } else {
fit_model <- brm(
    paste0(measure, "~ 1 + pmi + (1+pmi|pp_num) + (1+pmi|fan_word)"),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  prior = priors_full,
  control = list(adapt_delta = 0.95),
  iter = 3000,
  cores = 4
)
  }

# output of the model

fit_model

}

modeling_pmi <- function(useddata, measure, adaptdelta=FALSE) {

  if (adaptdelta==FALSE) {
fit_model <- brm(
    paste0(measure, "~ 1 + pmi + (1|pp_num) + (1|fan_word) "),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  save_all_pars = TRUE,
  prior = priors,
  iter = 10000,
  cores = 4
)
  } else {
fit_model <- brm(
    paste0(measure, "~ 1 + pmi + (1|pp_num) + (1|fan_word)"),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  prior = priors,
  save_all_pars = TRUE,
  control = list(adapt_delta = 0.95),
  iter = 10000,
  cores = 4
)
  }

# output of the model

fit_model

}

modeling_actr <- function(useddata, measure, adaptdelta=FALSE) {

  if (adaptdelta==FALSE) {
fit_model <- brm(
    paste0(measure, "~ 1 + actr + (1|pp_num) + (1|fan_word) "),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  save_all_pars = TRUE,
  prior = priors,
  iter = 10000,
  cores = 4
)
  } else {
fit_model <- brm(
    paste0(measure, "~ 1 + actr + (1|pp_num) + (1|fan_word)"),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  prior = priors,
  save_all_pars = TRUE,
  control = list(adapt_delta = 0.95),
  iter = 10000,
  cores = 4
)
  }

# output of the model

fit_model

}

## RT models

save_results <- function(measure, prior, bf, col_names=FALSE) {
data.frame(measure=measure, prior=prior, bf=bf) %>% rownames_to_column() %>%
    write_csv("bf-results.csv", col_names=col_names, append=TRUE)
}

#pmi_model_full <- modeling_pmi_full(merged_data, "rt", adaptdelta=TRUE)
#save(pmi_model_full, file="pmi_model_full.Rdata")
#print(pmi_model_full, digits=3)

fan_model <- modeling_fan(merged_data, "rt", adaptdelta=FALSE)
save(fan_model, file="../stan_models/fan_model.Rdata")
load("../stan_models/fan_model.Rdata")
print(fan_model, digits=3)

# pmi full has problems with convergence; we use the pmi model
pmi_model <- modeling_pmi(merged_data, "rt", adaptdelta=FALSE)
save(pmi_model, file="../stan_models/pmi_model.Rdata")
load("../stan_models/pmi_model.Rdata")
print(pmi_model, digits=3)

# actr full has problems with convergence; we use the actr model
actr_model <- modeling_actr(merged_data, "rt", adaptdelta=FALSE)
save(actr_model, file="../stan_models/actr_model.Rdata")
#load("../stan_models/actr_model.Rdata")
print(actr_model, digits=3)

for (i in 1:3) {
    
    BF_informative <- bayes_factor(pmi_model, fan_model)

    print(BF_informative)

    save_results("rt-pmi-fan", "normal(0,0.1)", BF_informative$bf)

  }

for (i in 1:3) {
    
    BF_informative <- bayes_factor(actr_model, fan_model)

    print(BF_informative)

    save_results("rt-actr-fan", "normal(0,0.1)", BF_informative$bf)

  }

for (i in 1:3) {
    
    BF_informative <- bayes_factor(pmi_model, actr_model)

    print(BF_informative)

    save_results("rt-pmi-actr", "normal(0,0.1)", BF_informative$bf)

  }

