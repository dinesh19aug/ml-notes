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





