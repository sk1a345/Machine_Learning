import pandas as pd
import numpy as np

# Loading the dataset:
df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\ml_regression_dataset.csv")
# print(df)

# Checking for the missing values:
# print(df.isnull().sum())


# Filling the missing values:
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='mean')

df = pd.DataFrame(si.fit_transform(df),columns=df.columns)
# print(type(df))
# print(df.isnull().sum())

# Removing the duplicates:
# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape)


# Diving the data into input and output data:
x= df.drop('Salary',axis=1)
y = df['Salary']

# Performing the scaling on the input data:
from sklearn.preprocessing import StandardScaler
si = StandardScaler()
x = pd.DataFrame(si.fit_transform(x),columns=x.columns)
# print(type(x))
# print(x)

# Diving the data into training and testing data:
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# print(x_train.shape)
# print(x.shape)
# print(x_test.shape)

# Visualizing the data:
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='Age',y='Experience',data=df,hue='Salary')
# plt.show()


# Training the model:
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=30)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

# Checking the accuray of the model:
from sklearn.metrics import mean_absolute_error,mean_squared_error
print("Mean absolute error: ",mean_absolute_error(y_test,y_pred))
print("Mean squared error:  ",mean_squared_error(y_test,y_pred))
print("Root mean squared error: ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("Score(train): ",knn.score(x_train,y_train)*100)
print("Score(test): ",knn.score(x_test,y_test)*100)

 
'''for i in range(1,20):
    knn1 = KNeighborsRegressor(n_neighbors=i)
    knn1.fit(x_train,y_train)
    y_pred1 = knn1.predict(x_test)
    print("Score(train): ",knn1.score(x_train,y_train)*100," Score(test): ",knn1.score(x_test,y_test)*100,"\n")
'''

# Taking the user input:
user_age = float(input("Enter your age: "))
user_exp = float(input("Enter your experience: "))

user_input = pd.DataFrame([[user_age,user_exp]],columns=x.columns)

user_input = pd.DataFrame(si.transform(user_input),columns=x.columns)
print(user_input)
user_pred_sal = knn.predict(user_input)
print("User salary: ",user_pred_sal[0])