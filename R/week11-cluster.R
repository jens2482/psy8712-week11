# Script Settings and Resources #Not sure where to put this comment so I'm putting it here. I had issues with pushing from MSI to github. Everything appeared to be linked up correctly and in MSI it told me that the commit and push went through, but it never appeared in GitHub. I'm hoping you'll be able to see that I did try it in the file showing you what I did in MSI. But I had to do some manual file transfers between MSI and my local device in order to make sure all the files were correctly saved.
library(tidyverse)
library(haven)
library(caret)
library(parallel) 
library(doParallel)
library(tictoc) 

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
train_gss_tbl <- as.data.frame(shuffled_data[1:split,]) 
test_gss_tbl <- as.data.frame(shuffled_data[(split+1):nrow(gss_tbl),] ) 

training_folds <- createFolds(train_gss_tbl$work_hours, 10) 

myControl <- trainControl( 
  method = "cv", 
  number = 10, 
  verboseIter = TRUE,
  indexOut = training_folds
)


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
toc_lm <- toc() 

tic() 
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

tic() 
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
toc_rf <- toc() 

tic() 
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
toc_xgb <- toc() 


local_cluster <- makeCluster(14) #doubled the amount from what I originally had (7)
registerDoParallel(local_cluster)

tic() 
model_lm <- train( 
  work_hours ~ ., 
  train_gss_tbl,
  method = "lm", 
  preProcess = c("center", "scale", "medianImpute", "nzv"), 
  na.action = na.pass, 
  trControl = myControl
)

p_lm <- predict(model_lm, 
                test_gss_tbl, 
                na.action = na.pass)
p_lm_results <- cor(p_lm, as.data.frame(test_gss_tbl)$work_hours)^2 
toc_lm_parallel <- toc() 

tic() 
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
toc_enet_parallel <- toc()

tic() 
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
toc_rf_parallel <- toc()

tic() 
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
toc_xgb_parallel <- toc() 

stopCluster(local_cluster)
registerDoSEQ()

# Publication
lm_train_results <- str_replace(formatC(max(model_lm$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "") 
lm_test_results <- str_replace(formatC(p_lm_results, format = "f", digits = 2), "^0", "") 
enet_train_results <- str_replace(formatC(max(model_enet$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "")
enet_test_results <- str_replace(formatC(p_enet_results, format = "f", digits = 2), "^0", "") 
rf_train_results <- str_replace(formatC(max(model_rf$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "") 
rf_test_results <- str_replace(formatC(p_rf_results, format = "f", digits = 2), "^0", "") 
xgb_train_results <- str_replace(formatC(max(model_xgb$results$Rsquared, na.rm = TRUE), format = "f", digits = 2), "^0", "") 
xgb_test_results <- str_replace(formatC(p_xgb_results, format = "f", digits = 2), "^0", "") 

"Table 3" <- tibble( 
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(lm_train_results, enet_train_results, rf_train_results, xgb_train_results),
  ho_rsq = c(lm_test_results, enet_test_results, rf_test_results, xgb_test_results)
)
write.csv(`Table 3`, "table3.csv")

"Table 4" <- tibble( 
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  supercomputer = c(toc_lm$callback_msg, toc_enet$callback_msg, toc_rf$callback_msg, toc_xgb$callback_msg),
  supercomputer_127 = c(toc_lm_parallel$callback_msg, toc_enet_parallel$callback_msg, toc_rf_parallel$callback_msg, toc_xgb_parallel$callback_msg)
)
write.csv(`Table 4`, "table4.csv")

#Which models benefited most from moving to the supercomputer and why?
  #Extreme Gradient Boost benefited the most from the supercomputer, especially when it comes to the parallelized models. There was a 88.81% decrease in the amount of time it took to run the model. It took a long time to run on my computer and almost no time on the supercomputer. I probably could have added more cores to make it go even faster too.
#What is the relationship between time and the number of cores used?
  #I wasn't exactly sure how many cores would work (the code making it the total number of cores minus 1 didn't work) so I went with 14, which is double the number of cores used on my computer. Between the non-parallelized models, there is about a correlation of about -0.10. Between the parallelized models in both the local and super computers the correlation is -0.47. So in all cases, the models decreased in total time elapsed when the supercomputer was used, but much more when the parallelization was utilized. 
#If your supervisor asked you to pick a model for use in a production model, would you recommend using the supercomputer and why? Consider all four tables when providing an answer.
  #Well for this task...I don't think the supercomputer was necessary. It does not change the R2 values or any other evaluation metrics. The only difference is the amount of time it takes. Just like you talked about how there's a certain threshold where creating a loop makes it worth it, I also feel that there would be a threshold under which a super computer just isn't necessary. For this task, I don't think I'd recommend the supercomputer, but if we were running models that took a few hours, the supercomputer may be worth it.