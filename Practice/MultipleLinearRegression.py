import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Practice/ml_regression_dataset.csv")
# print(df.head())
# print(df.shape)
# print(df.isnull().sum())

# Visualizing the data
sns.pairplot(data=df)
# plt.show()


# Filling missing values
for col in df.columns:
    df[col].fillna(df[col].mode()[0],inplace=True)


# print(df.isnull().sum())
# print(df.shape)

# Handelling outliers in dataset:

# print(df.describe())
# print(df.info())
# Visualizing outliers:

# sns.boxplot(x='Salary',data = df)
# sns.distplot(df['Salary'])
# sns.boxplot(x='Age',data = df)
# sns.distplot(df['Age'])
# sns.boxplot(x="Experience",data = df)
# sns.distplot(df['Experience'])
# plt.show()

# print(df.shape)
for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    min_range = q1-(1.5*iqr)
    max_range = q3+(1.5*iqr)
    df = df[(df[col]<=max_range) & (df[col]>=min_range)]

# print(df.shape)

# Dropping the duplicates:
df.drop_duplicates(inplace=True)
# print(df.shape)

# Feature scaling:
from sklearn.preprocessing import MinMaxScaler
# (apply only on the input features)
sc = MinMaxScaler()


# Splitting the data into training and testing data

x = df.drop('Salary',axis=1)
y = df['Salary']
print(x)
print(y)


x_scaled = sc.fit_transform(x)
x = pd.DataFrame(x_scaled,columns=x.columns)

# print(df.head())

# Visualizing the data
sns.pairplot(data=df)
# plt.show()



# Splitting the data into training and testing data

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# printing shapes:
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(x.shape)

# Applying the linear REgression algo

#accuracy_score is used for classification problems, not regression problems. For regression tasks, you should use metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or R² (coefficient of determination).
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


lr = LinearRegression() #creating the object:

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

acc = mean_squared_error(y_test,y_pred)

print("Accuracy score is: ",acc)

# Asking the user input:
user_age = float(input("Enter age: "))
user_expe = float(input("Enter experience: "))

user_input = np.array([[user_age,user_expe]])

# user_input_scaled = sc.transform(user_input) #directly passing numpy array

user_input_df = pd.DataFrame(user_input, columns=['Age', 'Experience']) #converting to dataframe
user_input_scaled = sc.transform(user_input_df)

sal_pred =lr.predict(user_input_scaled)

print(f"predicted salary is: {sal_pred[0]:2f}")




