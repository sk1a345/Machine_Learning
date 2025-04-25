import pandas as pd
import numpy as np

# Loading the data set:
df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\cgpa_package_dataset.csv")
# print(df.head())

df = df.drop("Student_ID",axis=1)
# print(df.head())

# Checking for the null values:
# print(df.isnull().sum()) #no null values present:

# Dropping the duplicates from the data:
# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape)

# Dividing the data into input and output data:
x = df.drop("Package",axis=1)
y = df['Package']
# print(x)
# print(y)

# Visualizing the data:
import matplotlib.pyplot as plt
import seaborn as sns
 
# sns.scatterplot(x='CGPA',y='Package',data=df)
# plt.show()

# Dividing the data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y.shape)
# print(y_train.shape)

# Training the model:
from sklearn.svm import SVR 
svr = SVR(kernel='poly')
svr.fit(x_train,y_train)
y_pred = svr.predict(x_test)
print("Score(train): ",svr.score(x_train,y_train)*100)
print("Score(test): ",svr.score(x_test,y_test)*100)


# Checking the accuracy of the model:
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("mean absolute error: ",mean_absolute_error(y_test,y_pred))
print("Mean Squared error: ",mean_squared_error(y_test,y_pred))
print("Root mean square error: ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 score: ",r2_score(y_test,y_pred))

# Visualing the data:
sns.scatterplot(x='CGPA',y='Package',data=df)
plt.scatter(df['CGPA'],svr.predict(x),color='red')
plt.show()
