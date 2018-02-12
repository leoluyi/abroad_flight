library(magrittr)
library(data.table)
library(caret)
library(mlbench)
library(tidyverse)
library(dtplyr)
# library(doParallel)
library(doSNOW)
library(rpart.plot)
library(MLmetrics)
library(InformationValue)

# Read data ---------------------------------------------------------------

dt_all_temp4 <- fread("./data/data_cleaned.csv", header = TRUE, check.names = F)

dt_all <- dt_all_temp4[, -c("CUST_ID", "YYYYMM", "MM", "age")]
dt_all[, y2 := y2 %>% as.factor]
dt_all %>% setnames(make.names(names(dt_all)))

# Slice data --------------------------------------------------------------

in_train <- createDataPartition(y = dt_all[["y2"]], p = 0.7, list = FALSE)
dt_training <- dt_all[in_train,]
# subset spam data (the rest) to test
dt_testing <- dt_all[-in_train,]
# dimension of original and training dataset
rbind("original dataset" = dim(dt_all),
      "training set" = dim(dt_training),
      "testing set"  =dim(dt_testing))
#                    [,1] [,2]
# original dataset 848708   28
# training set     594097   28
# testing set      254611   28

# Rank Features By Importance ---------------------------------------------

# control <- trainControl(method="repeatedcv", allowParallel = TRUE)
# 
# cl = makeCluster(3)
# registerDoParallel(cl)

# # train the model
# fit_lvq <- train(y2 ~., data = dt_all, method = "lvq", 
#                  metric = "Kappa", trControl = control)
# 
# # estimate variable importance
# importance <- varImp(fit_lvq)
# # summarize importance
# print(importance)
# # plot importance
# plot(importance)


# Parallel ----------------------------------------------------------------

# Register cluster so that caret will know to train in parallel.
cl <- makeCluster(3)
registerDoSNOW(cl)

# Use the doSNOW package to enable caret to train in parallel.
# While there are many package options in this space, doSNOW
# has the advantage of working on both Windows and Mac OS X.

# Logistic Regression -----------------------------------------------------

train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3)

system.time({
  fit_glm <- train(y2 ~ ., 
                  data = dt_training,
                  method = "glm",
                  trControl = train_control,
                  metric = "Kappa")
  saveRDS(fit_glm, file = "./models/fit_glm_20171221.Rds")
})
fit_glm <- readRDS("./models/fit_glm_20171219.Rds")
pred_test <- predict(fit_glm, dt_testing, type = "prob")[["1"]]
pred_test_1 <- predict(fit_glm, dt_testing)
MLmetrics::Recall(y_pred = pred_test_1, y_true = dt_testing$y2)
MLmetrics::Accuracy(y_pred = pred_test_1, y_true = dt_testing$y2)
MLmetrics::F1_Score(y_pred = pred_test_1, y_true = dt_testing$y2)
InformationValue::AUROC(dt_testing$y2, pred_test) # AUC: 0.7305
InformationValue::plotROC(dt_testing$y2, pred_test)
InformationValue::ks_plot(dt_testing$y2, pred_test)

varImp(fit_glm$finalModel) %>% 
  as.data.table(keep.rownames = T) %>% 
  arrange(desc(Overall))
#                  rn     Overall
#  1:       X10000005 65.68458288
#  2:       n_country 63.52553247
#  3:        spending 53.83334025
#  4:       X00000445 40.64609883
#  5:       X10000001 35.01658695
#  6:       X00001185 23.89640117
#  7:       X10000008 23.67933310
#  8:       X10000007 20.97110160
#  9:       X00001184 20.26049200
# 10:       season.Q1 17.62092507
# 11: age_cut..20.27. 16.28356078
# 12: age_cut..27.35. 14.80852076
# 13: age_cut..35.45. 11.07555197
# 14: age_cut..45.55.  7.68917538
# 15:             edu  7.10527959
# 16: age_cut..55.65.  6.59266298
# 17:       season.Q2  4.50415329
# 18:       X10000003  4.06003068
# 19:          income  3.75381702
# 20:           sex.F  2.11673829
# 21:       X10000010  0.48170223
# 22:       X10000004  0.22189109
# 23:       X10000002  0.20231717
# 24:       season.Q3  0.01687139


# Train Model - rf --------------------------------------------------------

train_control <- trainControl(method = "repeatedcv")

system.time({
  fit_rf <- train(y2 ~ ., 
                  data = dt_training,
                  method = "rf",
                  trControl = train_control,
                  metric = "Kappa")
  save(fit_rf, file = "./models/fit_rf_20171219.Rds")
})
#   user  system elapsed 
# 32.376   3.116 156.700 


# Train Model - CART --------------------------------------------------------

train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")
# tune_grid <- expand.grid(cp = c(0.05, 0.075, 0.1))

system.time({
  fit_cart <- train(y2 ~ ., 
                    data = dt_training,
                    method = "rpart",
                    # tuneGrid = tune_grid,
                    trControl = train_control,
                    metric = "Kappa")
  saveRDS(fit_cart, file = "./models/fit_cart_20171219.Rds")
})
#   user  system elapsed 
# 32.376   3.116 156.700 

rpart.plot(fit_cart$finalModel, extra=104,
           branch.lty=3, shadow.col="gray", nn=TRUE)

# Evaluation
varImp(fit_cart$finalModel)
# rpart variable importance
#
#   only 20 most important variables shown (out of 27)
#                   Overall
# n_country        100.0000
# spending          95.5586
# X10000005         63.6478
# X00000445         36.6456
# X10000001         36.0620
# season.Q1         18.4646
# income             4.8801
# edu                2.8204
# X10000003          2.5770
# age_cut..35.45.    1.2271
# age_cut..45.55.    1.1996
# age_cut..55.65.    0.9617
# age_cut..27.35.    0.8106
# sex.F              0.7791
# sex.M              0.5798
# age_cut..65.Inf.   0.4518
# X00001185          0.4460
# season.Q3          0.3838
# X10000008          0.3607
# age_cut..20.27.    0.2714

fit_cart <- readRDS("models/fit_cart_20171219.Rds")
pred_test <- predict(fit_cart, dt_testing, type = "prob")[["1"]]
pred_test_1 <- predict(fit_cart, dt_testing)
MLmetrics::ConfusionMatrix(y_pred = pred_test_1, y_true = dt_testing$y2)
MLmetrics::Recall(y_pred = pred_test_1, y_true = dt_testing$y2, positive = "1")
MLmetrics::Accuracy(y_pred = pred_test_1, y_true = dt_testing$y2)
MLmetrics::F1_Score(y_pred = pred_test_1, y_true = dt_testing$y2, positive = "1")
InformationValue::plotROC(dt_testing$y2, pred_test)
InformationValue::ks_plot(dt_testing$y2, pred_test)


# XGBoost -----------------------------------------------------------------

# Leverage a grid search of hyperparameters for xgboost. See 
# the following presentation for more information:
# https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1
train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")

tune_grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)

# Train the xgboost model using 10-fold CV repeated 3 times 
# and a hyperparameter grid search to train the optimal model.

system.time({
  fit_xgb <- train(y2 ~ ., 
                   data = dt_training,
                   method = "xgbTree",
                   tuneGrid = tune_grid,
                   trControl = train_control,
                   metric = "Kappa")
  # saveRDS(fit_xgb, file = "./models/fit_xgb_20171219.Rds")
})
# The final values used for the model were
# nrounds = 100, 
# max_depth = 8, eta = 0.1, gamma = 0, colsample_bytree = 0.5,
# min_child_weight = 2 and subsample = 1.
#    user    system   elapsed 
# 407.096   350.096 56468.372 

fit_xgb <- readRDS("./models/fit_xgb_20171219.Rds")
pred_test <- predict(fit_xgb, dt_testing, type = "prob")[["1"]]
pred_test_1 <- predict(fit_xgb, dt_testing)
MLmetrics::ConfusionMatrix(y_pred = pred_test_1, y_true = dt_testing$y2) # 0.01228315
MLmetrics::Recall(y_pred = pred_test_1, y_true = dt_testing$y2, positive = "1")
MLmetrics::Accuracy(y_pred = pred_test_1, y_true = dt_testing$y2) # 0.9077966
MLmetrics::F1_Score(y_pred = pred_test_1, y_true = dt_testing$y2, positive = "1") # 0.02419154
InformationValue::plotROC(dt_testing$y2, pred_test)
InformationValue::ks_plot(dt_testing$y2, pred_test)
# AUC: 0.7511
InformationValue::optimalCutoff(dt_testing$y2, pred_test, optimiseFor = "Both")
# [1] 0.09779701
MLmetrics::ConfusionMatrix(y_pred = cut(pred_test, breaks = c(0, 0.09779701, 1), 
                                    include.lowest = T,
                                    labels = c("0", "1")), 
                           y_true = dt_testing$y2)
InformationValue::sensitivity(dt_testing$y2, pred_test, threshold = 0.09779701)
# Recall: 0.7511713
InformationValue::precision(dt_testing$y2, pred_test, threshold = 0.09779701)
# Precision 0.1757925


# Stop Cluster -----------------------------------------------------------

stopCluster(cl)

# Reference ---------------------------------------------------------------

# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
# https://www.kaggle.com/steves/xgboost-with-caret
# https://github.com/datasciencedojo/meetup/blob/master/intro_to_ml_with_r_and_caret/IntroToMachineLearning.R

