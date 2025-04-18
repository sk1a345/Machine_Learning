# Given probelm is of the supervised learning(since labled data(data containing output values(sales)) is given)
# And the problem is of the Regression(continuous values is given)
# Linear Regression:
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Advertising.csv")

# Procedure to find the best model for the linear regression:
# print(df)
# sns.pairplot(df)
# sns.lineplot(x = df['TV'],y = df['Sales']) #best data for the linear regression(since the relation of the tv with the sales is linear)
# sns.scatterplot(x = df['TV'],y = df['Sales'])
# plt.show()

from sklearn.model_selection import train_test_split
x = df.drop("Sales",axis=1)
y = df['Sales']
# print(x)
# print(y)
# x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.2,random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train) #training the model 
y_pred = lr.predict(x_test) #input data for testing

# print(y_pred)

from sklearn.metrics import r2_score, mean_absolute_error

error = mean_absolute_error(y_test,y_pred)
err1 = r2_score(y_test,y_pred)

# print(error)
# print(err1)

# Predictive system:
# print(df.head())


def predict_sales(tv_budget,radio_budget,newspaper_budget):
    features = np.array([[tv_budget,radio_budget,newspaper_budget]])
    results = lr.predict(features).reshape(1,-1)
    return results[0]

r1 = df.iloc[0]
print(r1)
tv_budget =  230.1
radio_budget = 37.8
newspaper_budget =  69.2
sales = predict_sales(tv_budget,radio_budget,newspaper_budget)
print(sales)


import pickle #chatgpt explain this please
pickle.dump(lr,open('linear.pkl','wb')) #write binary:





