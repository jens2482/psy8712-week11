# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel) #added this library for parallelization
library(doParallel) #added this library for parallelization
library(tictoc) #added this library to track time

# Data Import and Cleaning
data = read_sav("../data/GSS2016.sav")
gss_tbl <- data %>%
  mutate_all(~ifelse(.==0, NA, .)) %>%  
  drop_na(MOSTHRS) %>% 
  rename(work_hours= MOSTHRS) %>% 
  select (-c(HRS1, HRS2)) %>% 
  select(where(~mean(is.na(.)) < 0.75)) %>%  
  sapply(as.numeric) 
  
# Visualization
ggplot (gss_tbl, aes(work_hours))+
  geom_histogram() +
  ylab("Frequency") +
  xlab("Work Hours")

# Analysis
set.seed(54321)

rows <- sample(nrow(gss_tbl))
shuffled_data <-  gss_tbl[rows,]
split <- round(nrow(gss_tbl)*0.75)
train_gss_tbl <- as.data.frame(shuffled_data[1:split,]) #added as.data.frame because otherwise my createFolds wouldn't work
test_gss_tbl <- as.data.frame(shuffled_data[(split+1):nrow(gss_tbl),] ) #made this one a data frame too for consistency

training_folds <- createFolds(train_gss_tbl$work_hours, 10) #didn't have this in my original code, but you said that we needed to do this in your video

myControl <- trainControl( 
  method = "cv", 
  number = 10, 
  verboseIter = TRUE,
  indexOut = training_folds
)

#models without parallelization
tic() #start time
model_lm <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "lm", 
  preProcess = c("center", "scale", "medianImpute", "nzv"), #added center and scale
  na.action = na.pass, 
  trControl = myControl
)

p_lm <- predict(model_lm, 
                test_gss_tbl, 
                na.action = na.pass)
p_lm_results <- cor(p_lm, as.data.frame(test_gss_tbl)$work_hours)^2 
toc_lm <- toc() #end time

tic() #start time
model_enet <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "glmnet",
  preProcess = c("center", "scale", "medianImpute", "nzv"), 
  na.action = na.pass,
  trControl = myControl
)

p_enet <- predict(model_enet, 
                  test_gss_tbl,
                  na.action = na.pass)
p_enet_results <- cor(p_lm, as.data.frame(test_gss_tbl)$work_hours)^2
toc_enet <- toc() #end time

tic() #start time
model_rf <- train( 
  work_hours ~ .,  
  train_gss_tbl,
  method = "ranger", 
  preProcess = c("center", "scale", "medianImpute", "nzv"),
  na.action = na.pass,
  trControl = myControl
)

p_rf <- predict(model_rf, 
                test_gss_tbl,
                na.action = na.pass)
p_rf_results <- cor(p_rf, as.data.frame(test_gss_tbl)$work_hours)^2
toc_rf <- toc() #end time

tic() #start time
model_xgb <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "xgbLinear", 
  preProcess = c("center", "scale", "medianImpute", "nzv"),
  na.action = na.pass,
  trControl = myControl
)

p_xgb <- predict(model_xgb, 
                  test_gss_tbl,
                  na.action = na.pass)
p_xgb_results <- cor(p_xgb, as.data.frame(test_gss_tbl)$work_hours)^2
toc_xgb <- toc() #end time

#Models with parallelization
local_cluster <- makeCluster(detectCores() - 1) #make cluster and set the number of cores to 1 less than number on device
registerDoParallel(local_cluster)

tic() #start time
model_lm <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "lm", 
  preProcess = c("center", "scale", "medianImpute", "nzv"), #added center and scale
  na.action = na.pass, 
  trControl = myControl
)

p_lm <- predict(model_lm, 
                test_gss_tbl, 
                na.action = na.pass)
p_lm_results <- cor(p_lm, as.data.frame(test_gss_tbl)$work_hours)^2 
toc_lm_parallel <- toc() #end time

tic() #start time
model_enet <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "glmnet",
  preProcess = c("center", "scale", "medianImpute", "nzv"), 
  na.action = na.pass,
  trControl = myControl
)

p_enet <- predict(model_enet, 
                  test_gss_tbl,
                  na.action = na.pass)
p_enet_results <- cor(p_lm, as.data.frame(test_gss_tbl)$work_hours)^2
toc_enet_parallel <- toc() #end time

tic() #start time
model_rf <- train( 
  work_hours ~ .,  
  train_gss_tbl,
  method = "ranger", 
  preProcess = c("center", "scale", "medianImpute", "nzv"),
  na.action = na.pass,
  trControl = myControl
)

p_rf <- predict(model_rf, 
                test_gss_tbl,
                na.action = na.pass)
p_rf_results <- cor(p_rf, as.data.frame(test_gss_tbl)$work_hours)^2
toc_rf_parallel <- toc() #end time

tic() #start time
model_xgb <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "xgbLinear", 
  preProcess = c("center", "scale", "medianImpute", "nzv"),
  na.action = na.pass,
  trControl = myControl
)

p_xgb <- predict(model_xgb, 
                 test_gss_tbl,
                 na.action = na.pass)
p_xgb_results <- cor(p_xgb, as.data.frame(test_gss_tbl)$work_hours)^2
toc_xgb_parallel <- toc() #end time

stopCluster(local_cluster) #stop cluster
registerDoSEQ() #register a sequential parallel backend 

# Publication
lm_train_results <- str_replace(formatC(max(model_lm$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "") 
lm_test_results <- str_replace(formatC(p_lm_results, format = "f", digits = 2), "^0", "") 
enet_train_results <- str_replace(formatC(max(model_enet$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "")
enet_test_results <- str_replace(formatC(p_enet_results, format = "f", digits = 2), "^0", "") 
rf_train_results <- str_replace(formatC(max(model_rf$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "") 
rf_test_results <- str_replace(formatC(p_rf_results, format = "f", digits = 2), "^0", "") 
xgb_train_results <- str_replace(formatC(max(model_xgb$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "") 
xgb_test_results <- str_replace(formatC(p_xgb_results, format = "f", digits = 2), "^0", "") 

table1_tbl <- tibble( 
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(lm_train_results, enet_train_results, rf_train_results, xgb_train_results),
  ho_rsq = c(lm_test_results, enet_test_results, rf_test_results, xgb_test_results)
)

table2_tbl <- tibble( 
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"), #add row labels for each model
  original = c(toc_lm$callback_msg, toc_enet$callback_msg, toc_rf$callback_msg, toc_xgb$callback_msg), #pull time amounts from toc for non-parallelized
  parallelized = c(toc_lm_parallel$callback_msg, toc_enet_parallel$callback_msg, toc_rf_parallel$callback_msg, toc_xgb_parallel$callback_msg) #pull time amounts from toc for parallelized
)

#Which models benefited most from parallelization and why?
  #eXtreme Gradient Boosting benefited the most from parallelization (a rougly 32% decrease in time, with the next closest being random forest with a 27% decrease) because it is the most complex and had the most combinations of different hyperparameters to test. 
#How big was the difference between the fastest and slowest parallelized model? Why?
  #There were 198.54 seconds between the fastest and slowest parallelized models. That is a smaller spread that with the non-parallelized models (300.6 seconds). So clearly parallelization helps to improve the speed, especially with more complex models. However, the simpler models can only go so much faster than they did in the first place so the time doesn't change much for those and they continue to run quickly.
#If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
  #Similar to the last assignment, I would still choose Random Forest. It has the highest R2 value between all the models and also runs a good amount faster than eXtreme gradient boosting both when parallelizing and not.