import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Import the datset
dataset = pd.read_csv('50_Startups.csv')

### Break up in dependent and Independent variables
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 4].values


#Split the data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# Fitting the SVR Model to the dataset
from sklearn.svm import SVR

# Create your regressor here
regressor = SVR(kernel='linear')
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
print(y_pred)


plt.scatter(X_train[:,2], y_train, color='red')
plt.scatter(X_test[:,2], y_pred, color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))




