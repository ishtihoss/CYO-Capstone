# Loading Packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
library(tidyverse)

# Importing Data

KAG_conversion_data <- read.csv("C:/Users/user/Downloads/KAG_conversion_data.csv")

KAG_index <- createDataPartition(KAG_conversion_data$Approved_Conversion,times=1,p=.3,list = FALSE)
KAG <- KAG_conversion_data %>% slice(KAG_index)

KAG_Bin <- KAG %>% mutate(Sales=ifelse(Approved_Conversion>0,"Yes","No"))
  
# Creating Test and Train Data

train_index <- createDataPartition(KAG_Bin$Sales,times=1,p=.7,list=FALSE)
train_part <- KAG_Bin %>% slice(train_index)
test_part <- KAG_Bin %>% slice(-train_index)

#Listing models

models <- c("knn", "kknn", "rf",  "Rborist")

# Training Models

fits <- lapply(models, function(model){ 
  print(model)
  train(Sales~as.factor(gender)+as.factor(interest)+Impressions+Clicks+Spent+Total_Conversion, method = model, data = train_part)
})

names(fits) <- models

# Creating Predictions

fits_predicts <- sapply(fits, function(fits){
  predict(fits,test_part)
})

j <- seq(1:4)
confusionvector <- sapply(j, function(j){
  confusionMatrix(fits_predicts[,j],test_part$Sales)$overall["Accuracy"]
})
