# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read the dataset
dataset = pd.read_csv("50_Startups.csv")

#Divide the dataset in dependent and Independent variables
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



#Taking care of Categorical values.
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label_encoder = LabelEncoder();
X[:,3]=label_encoder.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X= oneHotEncoder.fit_transform(X).toarray()

#getting out of dummy variable trap
X = X[:,1:] # Select all the rows and all the columns starting fom index 1 onwards.
#Create training and test set
from  sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20,
                                                    train_size=0.80,
                                                    random_state=0)


#Check for missing data
null_columns=dataset.columns[dataset.isnull().any()]
t = dataset[null_columns].isnull().sum()

#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the results of the training set
y_pred = regressor.predict(X_test)  


print(regressor.coef_)
print(regressor.intercept_)

print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))

################# Backward elimination #####################
import  statsmodels.formula.api as smf

#Appending ones for constants
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
x_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = smf.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

### Removing index 2 as P>0.05 and is the highest P
x_opt = X[:,[0,1,3,4,5]]
regressor_OLS = smf.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

### Removing index 1 as P>0.05 and is the highest P
x_opt = X[:,[0,3,4,5]]
regressor_OLS = smf.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

### Removing index 1 as P>0.05 and is the highest P
x_opt = X[:,[0,3,5]]
regressor_OLS = smf.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

### Removing index 1 as P>0.05 and is the highest P
x_opt = X[:,[0,3]]
regressor_OLS = smf.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())






