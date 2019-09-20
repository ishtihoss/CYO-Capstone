# Loading Packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
library(tidyverse)

# Importing Data

KAG_conversion_data <- read.csv("C:/Users/ishtiaque/Downloads/KAG_conversion_data.csv")


KAG_Bin <- KAG_conversion_data %>% mutate(Sales=ifelse(Approved_Conversion>0,"Yes","No"))

# Creating Test and Train Data

train_index <- createDataPartition(KAG_Bin$Sales,times=1,p=.7,list=FALSE)
train_part <- KAG_Bin %>% slice(train_index)
temp <- KAG_Bin %>% slice(-train_index)
test_part <- temp %>% 
  semi_join(train_part, by = "interest")

#TuneGrid

tune_grid <- expand.grid(
  nrounds = 200,
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 10, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

# Training Models

fits <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions+Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control,
              tuneGrid = tune_grid)


# Creating Predictions

fits_predicts <- predict(fits,test_part)

confusionMatrix(as.factor(fits_predicts),as.factor(test_part$Sales))$overall["Accuracy"]

# Tuning for Gamma

tune_grid_1 <- expand.grid(
  nrounds = 200,
  eta = fits$bestTune$eta,
  max_depth = fits$bestTune$max_depth,
  gamma = c(0,5,10,15,20,25,30,35,40,45,50),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# New fit with improved Gamma Tune

fits_1 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions+Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control,
                tuneGrid = tune_grid_1)

fits_predicts_1 <- predict(fits_1,test_part)

confusionMatrix(as.factor(fits_predicts_1),as.factor(test_part$Sales))$overall["Accuracy"]

# Tuning for Learning Rate

tune_grid_2 <- expand.grid(
  nrounds = 200,
  eta = c(0.01, 0.015, 0.025, 0.03, 0.05, 0.1),
  max_depth = fits$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# New fit with improved learning rate

fits_2 <- train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions+Clicks+Spent+Total_Conversion, method = "xgbTree", data = train_part,trControl = tune_control,
                tuneGrid = tune_grid_2)

fits_predicts_2 <- predict(fits_2,test_part)

confusionMatrix(as.factor(fits_predicts_2),as.factor(test_part$Sales))$overall["Accuracy"]






