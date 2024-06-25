#------------------------------------------------- Libraries -----------------------------------------------------#

# importing Basic libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
# we use this library to fill the empty slotes in data
from sklearn.impute import SimpleImputer
#To encode independent variables
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
#To encode dependent variables 
from sklearn.preprocessing import LabelEncoder
#Data spliting libraries
from sklearn.model_selection import train_test_split
#feature Scaling
from sklearn.preprocessing import StandardScaler 
#------------------------------------------------- Code -----------------------------------------------------#

# import dataset using pandas 
# iloc is an pandas lib function which will help to copy data row - col wise to a veriable
dataset = pd.read_csv('Data.csv') 


# all the rows , and all the columns except last column --> matrix of freatures x "vector"
x = dataset.iloc[: , : -1].values


# all the rows of last column as dependency variable "vector"
y = dataset.iloc[ : , -1].values

# all the values with empty slotes gets filled with mean of that column 
imputer= SimpleImputer(missing_values = np.nan , strategy= 'mean') # making inputer to raplace nan with avg 
imputer.fit(x[:,1:3]) # telling inputor to work on all rows and rows from 1 to 3
x[:,1:3] = imputer.transform(x[:,1:3]) 

#encoading independent variables and dependent variables 


# transformers take 3 arguments inside '[()]'  
# 1 . kind of transformation ( encoder ) ;
# 2 . class name that does encoading ( OneHotEncoder )
# 3 . in new '[]' index of column to transform 

#remainder takes only one argument that is passthrough -- this will makesure allOther stays same 
#ct == categoriacal variables == independent variables

ct = ColumnTransformer(transformers=[('encoader',OneHotEncoder(),[0])], remainder= 'passthrough') 

# ct.fit_transform dont return in an array...but we always need the data in array format..we we use np.array() to convert the data into array
x = np.array(ct.fit_transform(x) )

#dependent variables encoding 
le = LabelEncoder()  
y = le.fit_transform(y) 

#feature scaling is done after spliting dataset to Training set and Test set
#categorical , dependent , sizeOfData split 
X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size=0.2, random_state=1) 


# print(X_train)
# print('-----------')
# print(X_test)
# print('-----------')
# print(Y_train)
# print('-----------')
# print(Y_test)
# print('-----------')
sc = StandardScaler() 
X_train[:,3:] = sc.fit_transform(X_train[ :, 3:]) 
X_test[:, 3:] = sc.fit_transform(X_test[:3:])



print(X_train) 
print(X_test) 
















