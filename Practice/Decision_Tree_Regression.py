import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\ml_regression_dataset.csv")
# print(df.head())

# printing the null values:
# print(df.isnull().sum())
# filling the null values:

# using the simple imputer:
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='most_frequent')

df = pd.DataFrame(si.fit_transform(df),columns=df.columns)
# print(type(df))
# print(df.isnull().sum())

# REmoving the duplicates:
# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape)

# Diving data into input and output:
x = df.drop("Salary",axis=1)
y = df['Salary']

# SCaling the input values:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x),columns=x.columns)
# print(x)
# print(y)

# Dividing data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# Training the model using the Decison tree regression algo:
from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor()
dr.fit(x_train,y_train)

y_pred = dr.predict(x_test)

# Checking the accuracy:
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("Mean absolute error: ",mean_absolute_error(y_test,y_pred))
print("Mean squared erro: ",mean_squared_error(y_test,y_pred))
print("Root mean square error : ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 score: ",r2_score(y_test,y_pred))
print("Score: ",dr.score(x_test,y_test))

# Plotting the tree:
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(15,8))
# plot_tree(dr,filled=True,feature_names=x.columns)
# plt.show()

# Plotting the graph:

plt.scatter(x_test['Age'],y_test,color='blue',label='Actual')
plt.scatter(x_test['Age'],y_pred,color="Red",label="Predicted")
plt.xlabel("Features")
plt.ylabel("Salary")
plt.legend()
plt.title("Actual vs Predicted")
# plt.show()

# user input:
user_age = float(input("Enter the age: "))
user_expe = float(input("Enter the experience: "))

user_input = np.array([user_age,user_expe])
user_scaled_input = pd.DataFrame(sc.transform([user_input]),columns=x.columns)

user_sal_pred= dr.predict(user_scaled_input)
print("User's predicted salary is: ",user_sal_pred)
