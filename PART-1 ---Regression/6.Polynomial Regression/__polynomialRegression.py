#polynomial regression to find the previous job salary of a person
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

dataset = pd.read_csv('Position_Salaries.csv') 
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[ : , -1].values

# just a normal linear regression model on the data-set
linearRegression_1 = LinearRegression()
linearRegression_1.fit(x, y) 

#polynomial regression on data
polyRegression = PolynomialFeatures(degree = 4) 
x_poly = polyRegression.fit_transform(x)
#new x_poly is formed from x which is polynomial func with 3 features -- drawing a linearRegression on that new variable
linearRegression_2 = LinearRegression()
linearRegression_2.fit(x_poly , y) 

#drawing all the graphs
#normal linear regression
plt.scatter(x,y, color = 'red')
plt.plot(x,linearRegression_1.predict(x) , color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel = ('Position Level')
plt.ylabel = ('salary')
# plt.show() 

# polynomial data set graph
plt.scatter(x, y , color = 'red')
plt.plot(x,linearRegression_2.predict(polyRegression.fit_transform(x)) , color = 'blue')
plt.title('Truth or Bluff (polynomial Regression)')
plt.xlabel = ('Position Level')
plt.ylabel = ('salary')
# plt.show() 


print(linearRegression_1.predict([[6.5]]))
print(linearRegression_2.predict(polyRegression.fit_transform([[6.5]])))














