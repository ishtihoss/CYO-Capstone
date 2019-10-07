### Loading Necessary Packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(cloudml)) install.packages("cloudml")
if(!require(readr)) install.packages("readr")
if(!require(xgboost)) install.packages("xgboost")
if(!require(tinytex)) install.packages("tinytex")


library(tidyverse)
library(caret)
library(cloudml)
library(xgboost)
library(readr)
library(tinytex)

### Data Loading and Setup


data_dir <- gs_data_dir_local("gs://birds-nest-bb")
KAG_conversion_data <- read_csv(file.path(data_dir, "KAG_conversion_data.csv"))

### Add column for binary classification
KAG_Bin <- KAG_conversion_data %>% mutate(Sales=ifelse(Approved_Conversion>0,"Yes","No"))


#Now we move on to creating test and train sets:
  
set.seed(1989)
train_index <- createDataPartition(KAG_Bin$Sales,times=1,p=.7,list=FALSE)
train_part <- as.tbl(KAG_Bin) %>% dplyr::slice(train_index)
temp <- as.tbl(KAG_Bin) %>% dplyr::slice(-train_index)

# Making sure that the interests appearing in the test set are also in the train set
test_part <- temp %>% 
  semi_join(train_part, by = "interest")


### Results from XGBoost Default Tune

fits_0 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions
                +Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part)


# Creating Predictions

fits_0_predicts <- predict(fits_0,test_part)

confusionMatrix(as.factor(fits_0_predicts),as.factor(test_part$Sales),positive = "Yes")

default_tune <- sensitivity(as.factor(fits_0_predicts),as.factor(test_part$Sales),positive = "Yes")


### Choosing Parameters for crossvalidation and XGBosst tuning grid 

# At first we use a broad range of values to optimize our hyperparameters for XGboost

# Trying Broad Range of Values for XGBoost Tuning Grid
tune_grid <- expand.grid(
  nrounds = seq(from = 10, to = 200, by = 10),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Cross-validation

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 10, # with n folds 
  verboseIter = FALSE, # no training log
  allowParallel = FALSE # FALSE for reproducible results 
)


### Training Model on Broad Range of Hyperparameters (Step 1)


# Using the dimensions that are most relevant and variable 

fits <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions
              +Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control, tuneGrid = tune_grid)


# Looking at best tuning values
fits$bestTune

# Creating Predictions

fits_predicts <- predict(fits,test_part)

confusionMatrix(as.factor(fits_predicts),as.factor(test_part$Sales),positive = "Yes")

first_tune <- sensitivity(as.factor(fits_predicts),as.factor(test_part$Sales),positive = "Yes")

### Tuning for maxdepth and minimum child weight (Step 2)


tune_grid_2 <- expand.grid(
  nrounds = seq(from = 10, to = 200, by = 10),
  eta = fits$bestTune$eta,
  max_depth = seq(1:6),
  gamma = fits$bestTune$gamma,
  colsample_bytree = 1,
  min_child_weight = seq(1:3),
  subsample = 1
)

fits_2 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions
                +Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control,
                tuneGrid = tune_grid_2)

fits_predicts_2 <- predict(fits_2,test_part)

confusionMatrix(as.factor(fits_predicts_2),as.factor(test_part$Sales),positive = "Yes")
second_tune <- sensitivity(as.factor(fits_predicts_2),as.factor(test_part$Sales),positive = "Yes")


fits_2$bestTune

### Tuning for column and row sampling (Step 3)

tune_grid_3 <- expand.grid(
  nrounds = seq(from = 10, to = 200, by = 10),
  eta = fits$bestTune$eta,
  max_depth = fits_2$bestTune$max_depth,
  gamma = fits$bestTune$gamma,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = fits_2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

fits_3 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions
                +Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control, 
                tuneGrid = tune_grid_3)

fits_predicts_3 <- predict(fits_3,test_part)

confusionMatrix(as.factor(fits_predicts_3),as.factor(test_part$Sales),positive = "Yes")

third_tune <- sensitivity(as.factor(fits_predicts_3),as.factor(test_part$Sales),positive = "Yes")

fits_3$bestTune

### Tuning for Regularization (Step 4)

tune_grid_4 <- expand.grid(
  nrounds = seq(from = 10, to = 200, by = 10),
  eta = fits$bestTune$eta,
  max_depth = fits_2$bestTune$max_depth,
  gamma = c(0,5,10,15,20,25,30,35,40,45,50),
  colsample_bytree = fits_3$bestTune$colsample_bytree,
  min_child_weight = fits_2$bestTune$min_child_weight,
  subsample = fits_3$bestTune$subsample
)

# New fit to detect best value for gamma

fits_4 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions
                +Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control,
                tuneGrid = tune_grid_4)

fits_predicts_4 <- predict(fits_4,test_part)

confusionMatrix(as.factor(fits_predicts_4),as.factor(test_part$Sales),positive = "Yes")

fourth_tune <- sensitivity(as.factor(fits_predicts_4),as.factor(test_part$Sales),positive = "Yes")


fits_4$bestTune

###Optimizing learning rate (Step 5)


# Tuning for Learning Rate

tune_grid_5 <- expand.grid(
  nrounds = seq(from = 10, to = 2000, by = 10),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = fits_2$bestTune$max_depth,
  gamma = fits_4$bestTune$gamma,
  colsample_bytree = fits_3$bestTune$colsample_bytree,
  min_child_weight = fits_2$bestTune$min_child_weight,
  subsample = fits_3$bestTune$subsample
)

# New fit with improved learning rate

fits_5 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions
                +Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control,
                tuneGrid = tune_grid_5)

fits_predicts_5 <- predict(fits_5,test_part)

confusionMatrix(as.factor(fits_predicts_5),as.factor(test_part$Sales),positive = "Yes")

fifth_tune <- sensitivity(as.factor(fits_predicts_5),as.factor(test_part$Sales),positive = "Yes")


fits_5$bestTune

# Fitting the final model


final_grid <- expand.grid(
  nrounds = fits_5$bestTune$nrounds,
  eta = fits_5$bestTune$eta,
  max_depth = fits_2$bestTune$max_depth,
  gamma = fits_4$bestTune$gamma,
  colsample_bytree = fits_3$bestTune$colsample_bytree,
  min_child_weight = fits_2$bestTune$min_child_weight,
  subsample = fits_3$bestTune$subsample
)

# Final Fit

fits_final <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)
                    +Impressions+Clicks+Spent+Total_Conversion, 
                    method = "xgbTree", data = train_part,trControl = tune_control,
                    tuneGrid = final_grid)

fits_final_pred <- predict(fits_final,test_part)

confusionMatrix(as.factor(fits_final_pred),as.factor(test_part$Sales),positive = "Yes")

final_tune <- sensitivity(as.factor(fits_final_pred),as.factor(test_part$Sales),positive = "Yes")


# Trying with a larger cross-validation set

tune_control_f <- caret::trainControl(
  method = "cv", # cross-validation
  number = 25, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = FALSE # FALSE for reproducible results 
)

fits_final_f <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)
                      +Impressions+Clicks+Spent+Total_Conversion, 
                      method = "xgbTree", data = train_part,
                      trControl = tune_control_f,
                      tuneGrid = final_grid)

fits_final_pred_f <- predict(fits_final_f,test_part)

confusionMatrix(as.factor(fits_final_pred_f),as.factor(test_part$Sales),positive = "Yes")

final_tune_with_larger_CV <- sensitivity(as.factor(fits_final_pred_f),as.factor(test_part$Sales),positive = "Yes")

# Trying out the ensemble method 

#Model Stacking

models <- c("kknn","adaboost","rf","Rborist")

# Training HULK model


fits_ensemble <- lapply(models, function(model){ 
  print(model)
  train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+
          Impressions+Clicks+Spent+Total_Conversion, 
        method = "xgbTree", data = train_part)
}) 

names(fits_ensemble) <- models

# Creating a matrix of predictions

fits_ensemble_pred <- sapply(fits_ensemble, function(fits){
  predict(fits,test_part)
})

total_fits <- as.data.frame(fits_ensemble_pred) %>% mutate(xgbTree=fits_final_pred_f)

# Using majority voting method

df <- data.frame(matrix(unlist(total_fits), nrow=length(total_fits), byrow=T))
colnames(df) <- seq(1:nrow(total_fits))
rownames(df) <- c("kknn","adaboost","rf","Rborist","xgbTree")

col_index <- seq(1,ncol(df), 1)
predict_vote <- map_df(col_index, function(j){
  vote <- ifelse(test = sum(df[,j] == "Yes") > 3, yes = "Yes", no = "No")
  return(tibble(vote = vote))
})   # returns a df

predict_vote <- as.factor(predict_vote$vote) #  as factor

confusionMatrix(factor(predict_vote),  factor(test_part$Sales),positive = "Yes")

ensemble_method <- sensitivity(factor(predict_vote),  factor(test_part$Sales),positive = "Yes")

# Results 


results <- data.frame(Model_Name=c("Default Tune","First Tune","Second Tune","Third Tune","Fourth Tune","Fifth Tune","Final Tune","Final Tune with 25% CV", "Ensemble Majority Vote"),Sensitivity=c(default_tune,first_tune,second_tune,third_tune,fourth_tune,fifth_tune,final_tune,final_tune_with_larger_CV,ensemble_method))


results %>% ggplot(aes(x=reorder(Model_Name,Sensitivity),Sensitivity)) + geom_bar(stat="identity",fill="green")+coord_flip(y=c(0,1)) + labs(title = "Sensitivity by Model",x="Model Name",y="Sensitivity")+geom_text(aes(label=Sensitivity))     










