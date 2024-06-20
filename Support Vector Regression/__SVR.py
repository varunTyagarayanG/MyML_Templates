import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  
#polynomial libraries
from sklearn.preprocessing import PolynomialFeatures
#SVR 
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv') 
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[ : , -1].values

#Standard Scalar function needs 2d array as input
#no of rows ans no of columns 
y = y.reshape(len(y) ,1) 


#feature Scalling 
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler() 
y =sc_y.fit_transform(y) 

regressior = SVR(kernel= 'rbf') 
regressior.fit(x , y) 
sc_y.inverse_transform(regressior.predict(sc_x.transform([[6.5]])).reshape(-1,1)) 

plt.scatter(sc_x.inverse_transform(x) , sc_y.inverse_transform(y) , color = 'red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressior.predict(x).reshape(-1,1)) , color = 'blue')
plt.title('Truth or Bluff (SVR-- Regression)')
plt.xlabel = ('Position Level')
plt.ylabel = ('salary')
plt.show() 



