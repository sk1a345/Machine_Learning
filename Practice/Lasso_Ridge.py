import pandas as pd
import numpy as np

# Loading the dataset:
df = pd.read_csv("Practice/house_price_data.csv")
# print(df)


# Checking null values:
# print(df.isnull().sum())

# print(df.shape) #checking shape before dropping the duplicates:

# Dropping duplicates:
df.drop_duplicates(inplace=True)

# print(df.shape) #checking shape after dropping the duplicates:

# Visualizing the data using the heatmap

import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize=(10,7))
# sns.heatmap(df)
# plt.show()

# print(df.describe())

# checking for the outliers in the area column:

# using the boxplot:
# sns.boxplot(x='area',data = df)
# using the histplot
# sns.histplot(df['area'])
# plt.show()


# Input and output features:
x = df.drop('price',axis=1)
y = df['price']

# Performing Scing on the input columns:
# from sklearn.preprocessing import MinMaxScaler

# sc = MinMaxScaler()
# x = pd.DataFrame(sc.fit_transform(x),columns=x.columns)
# print(df)

# Splitting the data into the training and testing data:

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# Applying the Linear REgression to check the model Accuray and the cost function:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\nLinear Regression: ")
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

# Mean absolute error
mae = mean_absolute_error(y_test,y_pred)
print("Mean absolute error is: ",mae)

# Mean Squared error:

mse = mean_squared_error(y_test,y_pred)
print("Printing the mean square error: ",mse)

# Root mean square error:

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Printing the root mean square error: ",rmse)

# # Printing the score:

score = lr.score(x_test,y_test)
print("Score is: ",score)

# print(type(x))  # <class 'pandas.core.frame.DataFrame'>
# print(type(y))  # <class 'pandas.core.series.Series'>

# Visualizing:This visualizes the coefficients (also called weights) learned by the Linear Regression model.
plt.figure(figsize=(10,7))
plt.bar(x.columns,lr.coef_)
plt.title("Linear Regression")
plt.xlabel("columns")
plt.ylabel("Coef")
plt.show()

# Visualizing the actual price vs predicted price of the model:
sns.scatterplot(x = y_test,y = y_pred)
plt.title("Linear Regression Actual price vs Predicted price")
plt.xlabel("Acutal price")
plt.ylabel("predicted price")
plt.show()


# Training the model on Lasso Regression:
print("\nLasso: ")
from sklearn.linear_model import Lasso

ls = Lasso(alpha=0.1)

ls.fit(x_train,y_train)
y_pred1 = ls.predict(x_test)

# Mean absolute error:
mae1 = mean_absolute_error(y_test,y_pred1)
print(" Mean absolute error: ",mae1)

# Mean square error:
mse1 = mean_squared_error(y_test,y_pred1)
print("Mean square error: ",mse1)

#root Mean square error:
rmse1 = np.sqrt(mean_squared_error(y_test,y_pred1))
print("Root mean square is: ",rmse1)

# Printing teh score:
score1 = ls.score(x_test,y_test)
print("Score1: ",score1)

# Visualizing the graph:
plt.figure(figsize=(10,7))
plt.bar(x.columns,ls.coef_)
plt.title("Lasso Regression")
plt.xlabel("Columns")
plt.ylabel("Coefficient")
plt.show()

# Visualizing the actual price vs predicted price of the model:
sns.scatterplot(x = y_test,y = y_pred1)
plt.title("Lasso Regression Actual price vs Predicted price")
plt.xlabel("Acutal price")
plt.ylabel("predicted price")
plt.show()

# Ridge Regression:
print("\nRidge: ")
from sklearn.linear_model import Ridge
ri = Ridge(alpha=0.1)

ri.fit(x_train,y_train)
y_pred2 = ri.predict(x_test)

# Mean absolute error:
mae2 = mean_absolute_error(y_test,y_pred2)
print("Mean absolute eror : ",mae2)

# Mean square error:
mse2 = mean_squared_error(y_test,y_pred2)
print("Mean squared erro: ",mse2)

# root mean square error:
rmse2 = np.sqrt(mean_squared_error(y_test,y_pred2))
print("Root mean square error : ",rmse2)

# Score:
score2 = ri.score(x_test,y_test)
print("Score: ",score2)

# Visualizing:
plt.figure(figsize=(10,7))
plt.bar(x.columns,ri.coef_)
plt.title("Ridge Regression")
plt.xlabel("Columns")
plt.ylabel("Coefficients")
plt.show()

# Visualizing the actual price vs predicted price of the model:
sns.scatterplot(x = y_test,y = y_pred2)
plt.title("Ridge Regression Actual price vs Predicted price")
plt.xlabel("Acutal price")
plt.ylabel("predicted price")
plt.show()