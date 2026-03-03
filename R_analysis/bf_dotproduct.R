library(tidyverse)
library(tidybayes)
library(here)
library(brms)
library(rstan)
library(polspline)
testing_all <- read_csv("../csv_files/online_testing_all.csv")

testing_all$charcount <- nchar(testing_all$test_sentence)
dotproduct <- read_csv("../csv_files/dotproducts_perword.csv")
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

# We subset by target (because these dot products were calculated on trained target-context pairs, 
# and only targets were trained).

merged_data <- subset(merged_data, condition == "target")

full_priors <- list(c(
  set_prior("normal(7, 3)", class = "Intercept"),
  set_prior("halfnormal(0, 0.025)", class = "b"),
  set_prior("normal(0, 1)", class = "sigma"),
  set_prior("normal(0, 1)", class = "sd"),
  set_prior("lkj(2)", class = "cor")
),
c(
  set_prior("normal(7, 3)", class = "Intercept"),
  set_prior("halfnormal(0, 0.05)", class = "b"),
  set_prior("normal(0, 1)", class = "sigma"),
  set_prior("normal(0, 1)", class = "sd"),
  set_prior("lkj(2)", class = "cor")
))

stancode_full <- function(useddata, priors) {

stancode_model <- stancode(paste0("rt~ 1 + pmi + (1|pp_num) + (1+pmi|fan_word) "),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  save_all_pars = TRUE,
  prior = priors,
  warmup = 1000,
  iter = 2000,
  cores = 4
)

# the stancode

stancode_model

}

standata_full <- function(useddata, priors) {

standata_model <- standata(paste0("rt~ 1 + pmi + (1|pp_num) + (1+pmi|fan_word) "),# or (1|pers_word) + (1|loc_word)
  data = useddata,
  family = shifted_lognormal(),
  save_all_pars = TRUE,
  prior = priors,
  warmup = 1000,
  iter = 2000,
  cores = 4
)

# the standata

standata_model

}


## RT models

save_results <- function(measure, prior, bf, col_names=FALSE) {
data.frame(measure=measure, prior=prior, bf=bf) %>% rownames_to_column() %>%
    write_csv("bf-results-savage-dickey.csv", col_names=col_names, append=TRUE)
}

# priors: halfnormal with sd 0.025 and with sd 0.05
usedpriors <- c("halfnormal(0,0.025)", "halfnormal(0,0.05)")
priors_pointvalues <- c(2*dnorm(0, 0, 0.025), 2*dnorm(0, 0, 0.05))

j <- 1 # just one BF; more could be run if needed but we report only sd=0.025

#for (j in c(1, 2)) {

  mfull <- stan(paste0("../bf_models/modeling_bywordslopes_", usedpriors[j], ".stan"),
                      data = standata_full(merged_data, priors=full_priors[[j]]),
                      cores = 4,
                      warmup = 6000,
                      iter = 12000,
    control = list(adapt_delta = 0.99)
          )


  print(mfull, digits=3)

  b.posterior <- extract(mfull)$b

  fit.posterior <- logspline(b.posterior[,1], ubound=0)

  posterior <- dlogspline(0, fit.posterior)
  prior <- priors_pointvalues[j]
  BF01 <- posterior/prior
  print("BF10, b")
  print(1/BF01)
  
  save_results("rt-bywordslopes", usedpriors[j], 1/BF01)

#}
