import sklearn 
from sklearn import datasets
import pandas as pd
'''
try:
    print("scikit-learn is installed. Version:", sklearn.__version__)
except ImportError:  
    print("scikit-learn is NOT installed.") 

print("Using scikit-learn from:", sklearn.__file__)
print("Version:", sklearn.__version__) 
data = load_iris()
print(data) 
x, y = load_iris(return_x_y = True)'''
'''data = load_iris()
x = data.data
y = data.target

# print(x)  
# print(y)   
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# model.predict(x)
model.fit(x,y)
p = model.predict(x)
print(p)''''''
from sklearn.datasets import load_iris
x, y = load_iris(return_X_y = True)
print(x)
print(y)'''
'''
url = "https://raw.githubusercontent.com/611noorsaeed/100-days-Scikit-Learn-Tutorials-/refs/heads/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
d = pd.read_csv(url)
print(d)'''
'''
breast = datasets.load_breast_cancer()
# print(breast)
# print(breast.data)
# print(breast.target)
# x,y = datasets.load_iris(return_X_y = True)
# print(x)
# print(y)

df = pd.read_csv("CarDetails.csv")
# print(dpd)
x = df.drop("selling_price",axis=1)
y = df["selling_price"]
# print(x)
# print(x.shape)
# print(y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size = 0.2,random_state=1)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
print(x_train.shape)
print(x_test.shape)
print(x.shape)
'''

'''# Feature Scaling:
import pandas as pd
df = pd.read_csv("CarDetails.csv")
df = df[['selling_price','km_driven','year']] #selling_price =>output, km_driven,year =>input:

# print(df)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scald_df =scaler.fit_transform(df)
# print(scald_df) 

scald_df = pd.DataFrame(data=scald_df,columns=df.columns)
# print(scald_df)

from sklearn.model_selection import train_test_split 
x = df.drop('selling_price',axis=1)
y = df['selling_price']
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(x.shape)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

x_train_scalled = scaler.fit_transform(x_train)
x_test_scalled = scaler.transform(x_test)
# y_train_scalled = scaler.transform(y_train)
# y_test_scalled = scaler.transform(y_test)
print(x_train_scalled)
print(x_train)

x_train_scalled = pd.DataFrame(data = x_train_scalled, columns = x_train.columns)
x_test_scalled = pd.DataFrame(data = x_test_scalled,columns =x_train.columns)
# print(x_train_scalled)
# print(x_test_scalled)

# y_train_scalled = pd.DataFrame(data = y_train_scalled,columns = y_train.columns)
# y_test_scalled = pd.DataFarame(data = y_test_scalled ,columns = y_train.columns)
# print(y_train_scalled)
# print(y_test_scalled)
# _train is a pandas Series (1D), but MinMaxScaler expects a 2D array (like a column vector). So, this line:y_train_scalled = scaler.transform(y_train)
# Raises an error like:Expected 2D array, got 1D array instead

'''
'''
# https://github.com/611noorsaeed/100-days-Scikit-Learn-Tutorials-
# Standard_Scaler in scikit_learn:

import numpy as np #linear-algebra:
import pandas as pd #data processing:
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Social_Network_Ads.csv")
# print(df)

df.drop("User ID",axis = 1,inplace=True)
# print(df)

df["Gender"] = df["Gender"].map({"Male":1,"Female":0})
# print(df.head())

from sklearn.model_selection import train_test_split 

x = df.drop("Purchased",axis = 1)
y = df["Purchased"]
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state = 1)

print(x.shape)
print(x_train.shape)
print(y.shape)
print(y_train.shape)

from sklearn.preprocessing import StandardScaler 
scaler =  StandardScaler()

x_train_scalled = scaler.fit_transform(x_train)
x_test_scalled = scaler.fit_transform(x_test)
print(x_train_scalled)
# print(x_test_scalled)

print(np.round(x_train.describe(),2))
# print(x_train_scalled.describe())

x_train_scalled=pd.DataFrame(data = x_train_scalled,columns = x_train.columns)
x_test_scalled =pd.DataFrame(data = x_test_scalled,columns = x_train.columns)

print(x_train_scalled)
print(np.round(x_train_scalled.describe(),2))
'''

'''

# Label and Ordinal Encoder in ScikitLearn.(converting categorical data into numerical data)

import pandas as pd
import numpy as np


df = pd.read_csv('CarDetails.csv',usecols = ["name","seller_type","transmission","owner","fuel"])
print(df.head())

# x = df.drop("selling_price",axis=1)
# y = df["selling_price"]
df.drop("name",axis =1,inplace = True) #nominal data
print(df.head())

# USING PYTHON

# newdf = pd.DataFrame(data = df,columns = df.columns)
# print(newdf)

df["transmission"] = df["transmission"].map({"Manual":0,"Automatic":1})
# print(df)
# print(df["seller_type"].value_counts())
# df["transmission"] = df["transmission"].apply(lambda x:1 if x=="Manual" else 0) #using the lambda function:

df["seller_type"] = df["seller_type"].map({"Individual": 0,"Dealer":1,"Trustmark Dealer":2})
print(df)

print(df.sample(5))'''

'''# USING THE SCIKIT-LEARN


from sklearn.model_selection import train_test_split

x = df.drop("fuel",axis =1)
y = df["fuel"]
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(x.shape)
# LabelEncoder = output(y)
# OrdinalEncoder = input(x)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder #here LabelEncoder is the class

# output(y)
le = LabelEncoder() #creating the object
y_train_scaled =le.fit_transform(y_train)
y_test_scaled =le.transform(y_test)

# print(y_train_scaled)
# print(y_test_scaled)

# Input(x)
oe = OrdinalEncoder()
x_train_trans = oe.fit_transform(x_train)
x_test_trans = oe.transform(x_test)

print(x_train_trans)
print(x_test_trans)

'''

# One Hot Encoder and get_dummies:
import pandas as pd

df = pd.read_csv("CArDetails.csv")
# print(df)
get_new =pd.get_dummies(df.head(),columns=["name","fuel","owner"])
# print(get_new)
# using the scikit-learn:

from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder(categories = [['Mild','Strong']])
x_train_cough = ordinal.fit_tran


