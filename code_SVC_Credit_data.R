

# Support Vector Clasifier

library(ISLR)
library(e1071)
library(tidyverse)

set.seed(6822651)

#Loading data
data1= read.csv("Credit.csv")

#Create a binary variable
Y = ifelse(data1$Balance > median(data1$Balance), 1, 0)
data1$Balancebin= as.factor(Y)

#Fit a support vector classifier to the data with various values of cost, 
#in order to predict whether a customer gets high or low average credit card balance

svm.cv = tune(svm, Balancebin ~ ., data = data1, kernel = "linear", 
              ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100))) 
summary(svm.cv)

#the models Fitting
svm.lin = svm(Balancebin ~., data = data1, kernel = "linear", cost = 0.1)
summary(svm.lin)

#Creating confusions a matricend calculating error rates
Y.hat.lin = predict(svm.lin, data1)
conf.mat.lin = table(data1$Balancebin, Y.hat.lin)

#Confusion matrice
conf.mat.lin

### Calculating test error rates
err.lin = 1 - sum(diag(conf.mat.lin)) / sum(conf.mat.lin)
err.lin

