# Loading Packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
library(tidyverse)

# Importing Data

KAG_conversion_data <- read.csv("C:/Users/ishtiaque/Downloads/KAG_conversion_data.csv")

KAG_index <- createDataPartition(KAG_conversion_data$Approved_Conversion,times=1,p=.2,list = FALSE)
KAG <- KAG_conversion_data %>% slice(KAG_index)

KAG_Bin <- KAG %>% mutate(Sales=ifelse(Approved_Conversion>0,"Yes","No"))
  
# Creating Test and Train Data

train_index <- createDataPartition(KAG_Bin$Sales,times=1,p=.7,list=FALSE)
train_part <- KAG_Bin %>% slice(train_index)
temp <- KAG_Bin %>% slice(-train_index)
test_part <- temp %>% 
  semi_join(train_part, by = "interest")

#Listing models

models <- c("Rborist","rf","adaboost","xgbTree")

# Training Models

fits <- lapply(models, function(model){ 
  print(model)
  train(Sales~as.factor(age)+as.factor(gender)+as.factor(interest)+Impressions+Clicks+Spent+Total_Conversion, method = model, data = train_part)
})

names(fits) <- models

# Creating Predictions

fits_predicts <- sapply(fits, function(fits){
  predict(fits,test_part)
})

j <- seq(1:4)
confusionvector <- sapply(j, function(j){
  confusionMatrix(as.factor(fits_predicts[,j]),as.factor(test_part$Sales))$overall["Accuracy"]
})
