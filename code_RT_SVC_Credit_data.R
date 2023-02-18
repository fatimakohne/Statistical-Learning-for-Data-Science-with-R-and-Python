


# REGRESSION TREE

# Loading data
library(ISLR)
library(tree)
library(randomForest)
library(tidyverse)

set.seed(6822651)

#Loading data
credit_data= read.csv("Credit.csv")

# checking the missing value 
sum(is.na(credit_data)) # ' there is no missing value '

## Transform data
data1 <- credit_data[, 3:13]

### Transform variables
data1 <- data1 %>% 
  mutate(
    Gender = factor(Gender),
    Student = factor(Student),
    Married = factor(Married),
    Ethnicity = factor(Ethnicity),
  )

#Create training and test set
train =sample(1:nrow(data1), nrow(data1)*0.5) 
data.train = data1[train, ]
data.test = data1[-train, ]

#Fit a regression tree to the training set.
tree.credit =tree(Balance~., data = data.train)
summary(tree.credit)

#Plot the tree
plot(tree.credit)
text(tree.credit, pretty = 0)

#Calculating test-MSE
y.test = data.test$Balance
pred.credit = predict(tree.credit, data.test)
MSE.test = mean((y.test - pred.credit)^2)
MSE.test
((MSE.test)^0.5)

# Cross-validation
cv.credit = cv.tree(tree.credit, FUN = prune.tree)
par(mfrow = c(1,2))
plot(cv.credit$size, cv.credit$dev, type = "b")
plot(cv.credit$k, cv.credit$dev, type = "b")

# Re-fitting the tree with 5 nodes
prune.credit = prune.tree(tree.credit, best =5 )
plot(prune.credit)
text(prune.credit, pretty = 0)

# Use Cross-validation  in order to determine the optimal level of tree complexity
pred.prune = predict(prune.credit, data.test)
MSE.test.prune = mean((y.test - pred.prune)^2)
MSE.test.prune 
#Pruning the tree causes the test-MSE to increase to 46776.09

#Fit the model with Bagging
bag.credit = randomForest(Balance ~., data=data1, 
                          subset = train,mtry=10, importance=TRUE)
bag.credit


#The test set MSE using Bagging
pred.bag = predict(bag.credit, newdata = data.test)
MSE.test.bag = mean((pred.bag - y.test)^2)
MSE.test.bag



# SUPPORT VECTOR CLASSIFIER


library(e1071)

set.seed(6822651)

#Loading data
data2= read.csv("Credit.csv")

#Create a binary variable
Y = ifelse(data2$Balance > median(data2$Balance), 1, 0)
data2$Balancebin= as.factor(Y)

#Fit a support vector classifier 
svm.cv = tune(svm, Balancebin ~ ., data = data2, kernel = "linear", 
              ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100))) 
summary(svm.cv)

# Fit the model
svm.lin = svm(Balancebin ~., data = data2, kernel = "linear", cost = 0.1)
summary(svm.lin)

#Creating confusions a matricend calculating error rates
Y.hat.lin = predict(svm.lin, data2)
conf.mat.lin = table(data2$Balancebin, Y.hat.lin)

#Confusion matrice
conf.mat.lin

### Calculating test error rates
err.lin = 1 - sum(diag(conf.mat.lin)) / sum(conf.mat.lin)
err.lin









