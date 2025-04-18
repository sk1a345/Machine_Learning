import sklearn
from sklearn.datasets import load_iris
# print(load_iris()) #will load the data which is already present in the sklearn library
# print(load_iris(return_X_y=True)) #return input features and the output features:

x,y = load_iris(return_X_y=True) 
# print(x)
# print(y)

from sklearn.linear_model import LinearRegression
Model = LinearRegression()

# Fitting data:
Model.fit(x,y)
# print(Model.predict(x))

import pandas as pd
from sklearn.datasets import fetch_openml

df = fetch_openml('titanic',version=1,as_frame = True)['data']
# print(df.head())#first five rows
# print(df.shape)#shape
# print(df.info())#Information
# print(df.isnull().sum())#nullvalues

import seaborn as sns
import matplotlib.pyplot as plt

miss_val_percent = pd.DataFrame(df.isnull().sum()/len(df)*100)
miss_val_percent.plot(kind="bar",title="Missing values",ylabel='percentages')
# plt.show()

# Dropping the column:
# print(df.head)
df.drop(columns = ['body'],axis =1,inplace = True)
# print(df.shape)

# Value Imputation:
from sklearn.impute import SimpleImputer
print(f"No. of null values before imputing {df.age.isnull().sum()}")

imp = SimpleImputer(strategy = 'mean')
df['age'] =imp.fit_transform(df[['age']])

print(f"No of null values inside the age column after imputation: {df.age.isnull().sum()}")

# Data Encoding:

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
df[['female','male']] = ohe.fit_transform(df[['sex']])
print(df[['sex','female','male']])



