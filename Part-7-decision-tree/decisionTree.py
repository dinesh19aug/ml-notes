# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:10:34 2019

@author: dines
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("iris.csv")
X= dataset.iloc[0:, 0:4].values
Y = dataset.iloc[:,4]

setosa=dataset[dataset['species']=='Iris-setosa']
versicolor =dataset[dataset['species']=='Iris-versicolor']
virginica =dataset[dataset['species']=='Iris-virginica']

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))
plt.plot

setosa.plot(x="sepal-length", y="sepal-width", kind="scatter",ax=ax[0],label='setosa',color='r')
versicolor.plot(x="sepal-length",y="sepal-width",kind="scatter",ax=ax[0],label='versicolor',color='b')
virginica.plot(x="sepal-length", y="sepal-width", kind="scatter", ax=ax[0], label='virginica', color='g')

setosa.plot(x="petal-length", y="petal-width", kind="scatter",ax=ax[1],label='setosa',color='r')
versicolor.plot(x="petal-length",y="petal-width",kind="scatter",ax=ax[1],label='versicolor',color='b')
virginica.plot(x="petal-length", y="petal-width", kind="scatter", ax=ax[1], label='virginica', color='g')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()

plt.show()

from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
Y = labelEncoder_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0) 

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0,criterion='mse', splitter='best', max_depth=3)
regressor.fit(x_train, y_train)

y_predict=regressor.predict(x_test)

print(regressor.score(x_test,y_test))

pd.unique(Y)

from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree

dot_data = StringIO()  
tree.export_graphviz(regressor, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

