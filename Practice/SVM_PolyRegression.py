import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\Polynomial.csv")
# print(df.head())
# print(df.isnull().sum()) #no null values:


# splliting the data into input and output data:
x = df.drop("Salary",axis=1)
y = df['Salary']
# print(x.shape)
# print(y.shape)

# Visualzing the data:
import matplotlib.pyplot as plt
import seaborn as sns
# sns.scatterplot(x='Level',y='Salary',data =df)
# plt.show()

# Dividing the data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=45)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# Training the model:
from sklearn.svm import SVR
svr = SVR(kernel='linear')
svr.fit(x_train,y_train)
y_pred = svr.predict(x_test)
print("Score(training): ",svr.score(x_train,y_train)*100)
print("Score(Testing data): ",svr.score(x_test,y_test)*100)

# Checking the accuracy:
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("Absolute error: ",mean_absolute_error(y_test,y_pred))
print("Mean squared error: ",mean_squared_error(y_test,y_pred))
print("Root mean square error: ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 score: ",r2_score(y_test,y_pred))

# visulazing the data:
sns.scatterplot(x='Level',y='Salary',data = df)
plt.scatter(df['Level'],svr.predict(x),color='red')
plt.show()