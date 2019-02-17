# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 20:36:21 2019

@author: dines
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('BikeRental/bike_rental_train.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 3].values

#Split the data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=21)

#Fittiing the Simple linear regression on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=10 )
#regressor = LinearRegression( )
regressor.fit(X_train, y_train)

# Predicting the salary
y_pred = regressor.predict(X_test)

#Plot the data against training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Plot the data against test set
plt.scatter(X_test, y_test, color='red')
plt.scatter(X_test, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
