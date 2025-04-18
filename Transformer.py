import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer


df = pd.read_csv("Transform.csv")
# print(df)

x = df.drop("Strength",axis=1)
y = df['Strength']

# print(x)
# print(y)

# Train Test Split:
x = df.drop(columns=['Strength'])
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y.shape)
# print(y_train.shape)

# Check Data Distribution(in project use this code for data distribution)
# print(x_train.columns)
for col in x_train.columns:
    print(col)

# Without power Transformer:
# Applying regression without any transformation:
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
score = r2_score(y_test,y_pred)
print(score)

# Applying the Box_Cox Transform:
pt = PowerTransformer(method='box-cox')
x_train_trans = pt.fit_transform(x_train+0.00001)
x_test_transformed = pt.transform(x_test+0.00001)

# Applying linear regression on transformed data
lr = LinearRegression()
lr.fit(x_train_trans,y_train)
y_pred1 = lr.predict(x_test_transformed)
score1 = r2_score(y_test,y_pred1)
print(score1)

# Applying yeo-Johnson:

pt2 = PowerTransformer(method='yeo-johnson')
x_train_transformed = pt2.fit_transform(x_train)
x_test_trans = pt2.transform(x_test)

# Applying linear Regression on transformed data
lr = LinearRegression()
lr.fit(x_train_transformed,y_train)
y_pred2 = lr.predict(x_test_trans)
score2 = r2_score(y_test,y_pred2)
print(score2)