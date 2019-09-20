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