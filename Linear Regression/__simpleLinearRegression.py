#------------------------------------------------- Libraries -----------------------------------------------------#
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
#Regression Libraries
from sklearn.linear_model import LinearRegression
#------------------------------------------------- Code -----------------------------------------------------#

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size=0.2, random_state=0) 

#X_train == catagorrical data subPart for training 
#Y_train == dependent data SubPart for traning

#Starts here!
lr = LinearRegression()
regressior = LinearRegression() 
regressior.fit(X_train , Y_train)

y_pred = regressior.predict(X_test)

plt.scatter(X_train , Y_train , color ='red') 
plt.plot(X_train , regressior.predict(X_train),color = 'blue')
plt.title('salary Vs Experence 1')
plt.xlabel('Years Of Exp')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test , Y_test , color ='red') 
plt.plot(X_train , regressior.predict(X_train),color = 'blue')
plt.title('salary Vs Experence 2')
plt.xlabel('Years Of Exp')
plt.ylabel('Salary')

plt.show()








