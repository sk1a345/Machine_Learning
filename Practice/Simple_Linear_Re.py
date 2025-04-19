import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



df = pd.read_csv("practice/cgpa_package_dataset.csv")
print(df.head())

# Finding the missing values
# print(df.describe())

# Finding the null values inside the dataset:
# print(df.isnull().sum())

# Visualising the outliers
# Boxplot:
# sns.boxplot(x="Package",data = df)


# Checking whether graph is following linearity or not:
sns.scatterplot(x='CGPA',y="Package",data =df)
plt.show()
# Distribution plot:
# sns.histplot(df['Package'])
# plt.show()

sc = StandardScaler()
df['CGPA'] =sc.fit_transform(df[['CGPA']]) #usually scaling is performed on the input data
# print(df)


# print(df['Package'].mean())
# print(df['Package'].std())

# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape)

# Train-test split:

x = df.drop(['Package','Student_ID'],axis=1)
y = df['Package']



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
# print(x_train.shape)
# print(x_test.shape)
# print(x.shape)
# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
# print("Printing predicted value: ",y_pred)

# m = lr.coef_
# c = lr.intercept_
# y_pred1 = m*(x_test)+c
# print("Printing predicted value1: ",y_pred1)

acc = r2_score(y_test,y_pred)
print("Accuracy score is: ",acc)

user_input = float(input("Enter your CGPA score: "))
user_input = sc.transform([[user_input]])
# print(user_input)
predicted_package = lr.predict(user_input)
print(f"Predicted Package is: {predicted_package[0]:.2f} LPA")

# sns.scatterplot(x = 'CGPA',y = 'Package',data = df)
# plt.plot(df['CGPA'],lr.predict(x),c='red')
# plt.legend(['org data','predict line'])
# plt.show()


plt.scatter(x_test,y_test,color = 'blue',label='Actual Data')
plt.scatter(x_test,y_pred,color = 'red',label='Predicted Data')
plt.xlabel("CGPA(SCALED)")
plt.ylabel("Package")
plt.title("Actual vs Predicted Package")
plt.legend()
plt.show()