import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2:
data = {'Hours_study':[1,2,3,4,5,6,7,8,9,10],'Exam_score': [10,20,30,40,50,60,70,80,90,100]}

df = pd.DataFrame(data)
# print(df)

# Step 3:
# x = df.drop('Exam_score',axis=1)
# y = df['Exam_score']

x = df[['Hours_study']]
y = df[['Exam_score']]
# print(x)
# print(y)

# Step 5:
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)

# Step no. 6:
model = LinearRegression()
model.fit(x_train,y_train)

# Optional:
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
accu_score = r2_score(y_test,y_pred)
print(accu_score)

# User input testing:
user_input = float(input("Enter the no. of hours you study: "))
print(user_input)

pred_hr = model.predict([[user_input]])

print(f"User studied for {user_input} and scored : {pred_hr[0][0]:.2f} marks")

