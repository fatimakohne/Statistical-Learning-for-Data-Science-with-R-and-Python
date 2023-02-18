# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:37:24 2020

@author: Bryol Tene
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import sklearn.linear_model as skl_lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


#### read the data set 
data=pd.read_csv('C:/Users/Bryol Tene/Downloads/fish/Fish.csv')

### rename the variable 
data.rename(columns= {'Length1':'LengthVer', 'Length2':'LengthDia', 'Length3':'LengthCro'}, inplace=True)
print(data) 
                                                 
### remove the variable species then we dont wannt to have a categorial variable 
data1=data.drop(['Species'], axis=1)
print(data1) 

# there is no missing value. we can  startet directly with analyse 

###basic statistical tables/figures
data.describe().T
print(data.describe().T)

### Histogrmamme
data.hist(bins=5)

### coorlelation between die variable 
corr = data.corr()
print(corr)

#correlation Matrix (it has been removed from the pdf so as not to exceed the number of pages.  )
sns.heatmap(corr , annot=True, cmap='YlGnBu');

###  relation between the variable 
#Diese plot wurden nicht in die abgabe hinzugefügt, um die maximale Seiten nicht zu überschritten 

## relation between weight und Height
plt.scatter(data['Height'],data['Weight'], color = 'red')
plt.title('Height & Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.grid(True)

 ##relation between weight und LengthCro
plt.scatter(data['LengthCro'],data['Weight'], color = 'black')
plt.title('LengthCro  & Weight')
plt.xlabel('LengthCro')
plt.ylabel('Weight')
plt.grid(True)

## relation between weight und LengthVer
plt.scatter(data['LengthVer'],data['Weight'], color = 'indigo')
plt.title('LengthVer  & Weight')
plt.xlabel('LengthVer')
plt.ylabel('Weight')
plt.grid(True)

##relation between weight und LengthDia
plt.scatter(data['LengthDia'],data['Weight'], color = 'orange')
plt.title('LengthDia  & Weight')
plt.xlabel('LengthDia')
plt.ylabel('Weight')
plt.grid(True)

##relation between weight und Width
plt.scatter(data['Width'],data['Weight'], color = 'green')
plt.title('Width & Weight')
plt.xlabel('Width')
plt.ylabel('Weight')
plt.grid(True)



### simple lineare regresion 

## 
model1 = smf.ols(" Weight~ LengthVer ", data =data1).fit()
model1.summary()

##
model2 = smf.ols(" Weight~ LengthDia ", data =data1).fit()
model2.summary()

##
model3 = smf.ols(" Weight~ LengthCro ", data =data1).fit()
model3.summary()

##
model4 = smf.ols(" Weight~ Height ", data =data1).fit()
model4.summary()

##
model4 = smf.ols(" Weight~ Width ", data =data1).fit()
model4.summary()


### multiple linear regression 

model1 = smf.ols(" Weight~ LengthVer + LengthDia + LengthCro +  Height + Width", data =data1).fit()
model1.summary()

### fitted model plotten

##
sns.regplot(data['Height'],data['Weight'])
plt.title('Height & Weight')


##
sns.regplot(data['LengthVer'],data['Weight'])
plt.title('LengthVer & Weight')

##
sns.regplot(data['LengthDia'],data['Weight'])
plt.title('LenghtDia & Weight')

##
sns.regplot(data['LengthCro'],data['Weight'])
plt.title('LengthCro & Weight')

##
sns.regplot(data['Width'],data['Weight'])
plt.title('Width & Weight')



### model selection (Kfold cross validation)


#split the set of observations into two halves, by selecting a random subset of 79 observations out of the original 159 observations. We refer to these observations as the training set

train_df = data1.sample(79, random_state = 1)
test_df = data1[~data1.isin(train_df)].dropna(how = 'all')
X_train = train_df['LengthDia'].values.reshape(-1,1)
y_train = train_df['Weight']
X_test = test_df['LengthDia'].values.reshape(-1,1)
y_test = test_df['Weight']

# we use  LinearRegression()  to fit a linear regression to predict weight from LengthDia  using only the observations corresponding to the training set.

lm = skl_lm.LinearRegression()
model = lm.fit(X_train, y_train)

#We now use the  predict()  function to estimate the response for the test observations

pred = model.predict(X_test)
MSE = mean_squared_error(y_test, pred)
print(MSE)

## polynomial function 
#We can use the  PolynomialFeatures()  function to estimate the test error for the polynomial regressions

poly = PolynomialFeatures(degree=2)
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

model = lm.fit(X_train2, y_train)
print(mean_squared_error(y_test, model.predict(X_test2)))

## k fold cross validation
X = data1['LengthDia'].values.reshape(-1,1)
y = data1['Weight'].values.reshape(-1,1)
crossvalidation = KFold(n_splits=10, random_state=None, shuffle=False)

scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=crossvalidation,n_jobs=1)

print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))

for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=crossvalidation, n_jobs=1)
    print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))

