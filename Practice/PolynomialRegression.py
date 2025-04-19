import pandas as pd
import numpy as np


# Loading the dataset:
df = pd.read_csv('Practice/Polynomial.csv')
# print(df.head())

# Visualizing the dataset:
import matplotlib.pyplot as plt
import seaborn as sns
# plt.scatter(df['Level'],df['Salary'])
# plt.show()

# sns.scatterplot(x='Level',y = 'Salary',data = df)
# plt.show()

# Searching the null data:
# print(df.isnull().sum()) #no null found

# Removing the duplicates data:

# print(df.shape)
df.drop_duplicates(inplace=True) #no duplicates found:
# print(df.shape)

# Searching for the outliners in the data:
# using the boxplot:
# sns.boxplot(x='Level',y='Salary',data = df)
# using the distribution plot:
# sns.distplot(df['Level'])
# plt.show()

# print(df.describe())
# Removing the outliers in the salary column:
q1 = df['Salary'].quantile(0.25)
q3 = df['Salary'].quantile(0.75)

iqr = q3-q1
min_range = q1-(1.5*iqr)
max_range = q3+(1.5*iqr)

# print(df.shape)
df = df[(df['Salary']<=max_range) & (df['Salary']>=min_range)]
# print(df)
# print(df.describe())
# print(df.shape) #no outliers were detected

# Scaling the data using the Standard Scaler:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# print(df['Level'])
df['Level'] = sc.fit_transform(df[['Level']])

# print(df['Level'])
# print(df) 

# Splitting the data into training and testing data:

x = df[['Level']]
y = df['Salary']

# Converting to polynomial features:

from sklearn.preprocessing import PolynomialFeatures
pl = PolynomialFeatures(degree=2)   # Step 1: Initialize transformer for degree 2
pl.fit(x)                           # Step 2: Fit to the data (learn nothing, just prepares)
x = pl.transform(x)                # Step 3: Actually transforms your input x into polynomial features
# print(x)                           # Step 4: See the new transformed features

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# Training the model:
from sklearn.linear_model import LinearRegression


lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred =lr.predict(x_test)

print("Accuracy score: ",lr.score(x_test,y_test)) #It must be called with the test features (X) and the true values (y), so it can internally call .predict(X_test) and compare it to y_test.

# print(lr.coef_) #m1 =21427.80031224  m2 =-1161.04337871
# print(lr.intercept_) # c =43328.14289284563

# y1 = m1*x1+m2*(x2**2)+c

# Taking the user input:
# Scale the test value first
test_scaled = sc.transform([[8.116262258099887]])

# Then apply the polynomial transform
test_poly = pl.transform(test_scaled)

# Predict
test_pred = lr.predict(test_poly)
print(test_pred)
