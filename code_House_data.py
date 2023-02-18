# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:32:50 2020

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


### read  the data set 
data=pd.read_csv('C:/Users/Bryol Tene/Downloads/Houses.csv',index_col=[0])
# call the data
print(data) 


## there is no missing Value, we can startet directly with statistical analyse 


### basic statistical tables/figures

data.describe().T


### Histogrmamme
data.hist(bins=5)


### coorlelation between die variable 
corr = data.corr()
print(corr)

###correlation Matrix (it has been removed from the pdf so as not to exceed the number of pages.  )
sns.heatmap(corr , annot=True, cmap='YlGnBu');

###  relation between the variable (it has been removed from the pdf so as not to exceed the number of pages.  )

## relation between Price und Beds
plt.scatter(data['Beds'],data['Price'], color = 'black')
plt.title('Height & Price')
plt.xlabel('Beds')
plt.ylabel('Price')
plt.grid(True)


## relation between Price und Baths
plt.scatter(data['Baths'],data['Price'], color = 'black')
plt.title('Baths & Price')
plt.xlabel('Baths')
plt.ylabel('Price')
plt.grid(True)


## relation between Price und Size
plt.scatter(data['Size'],data['Price'], color = 'black')
plt.title('Size  & Price')
plt.xlabel('Size')
plt.ylabel('Price')
plt.grid(True)


## relation between Price und Lot
plt.scatter(data['Lot'],data['Price'], color = 'black')
plt.title('Lot & Price')
plt.xlabel('Lot')
plt.ylabel('Price')
plt.grid(True)



### simple linear regression

##
model1 = smf.ols(" Price~ Beds ", data =data).fit()
model1.summary()

##
model2 = smf.ols(" Price~ Baths ", data =data).fit()
model2.summary()

##
model3 = smf.ols(" Price~ Size ", data =data).fit()
model3.summary()

##
model4 = smf.ols(" Price~ Lot ", data =data).fit()
model4.summary()



## multiple linear regression 

model2 = smf.ols(" Price~ Beds + Baths + Size +  Lot ", data =data).fit()
model2.summary()


### Fitted model plotten 

##
sns.regplot(data['Beds'],data['Price'])
plt.title('Beds & Price')

##
sns.regplot(data['Baths'],data['Price'])
plt.title('Baths & Price')

##
sns.regplot(data['Size'],data['Price'])
plt.title('Size & Price')

##
sns.regplot(data['Lot'],data['Price'])
plt.title('Lot & Price')



####### model selection (K-fold cross Valdation)



#split the set of observations into two halves, by selecting a random subset of 26 observations out of the original 53 observations. We refer to these observations as the training set

train_df = data.sample(26, random_state = 1)
test_df = data[~data.isin(train_df)].dropna(how = 'all')
X_train = train_df['Size'].values.reshape(-1,1)
y_train = train_df['Price']
X_test = test_df['Size'].values.reshape(-1,1)
y_test = test_df['Price']

#We then use  LinearRegression()  to fit a linear regression to predict weight from LengthDia  using only the observations corresponding to the training set.

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
X = data['Size'].values.reshape(-1,1)
y = data['Price'].values.reshape(-1,1)
crossvalidation = KFold(n_splits=10, random_state=None, shuffle=False)

scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=crossvalidation,n_jobs=1)

print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))

for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=crossvalidation, n_jobs=1)
    print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))



