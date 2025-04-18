# A telecommunications company wants to reduce customer churn by identifying customers at risk of leaving they have historical data on customer behavior and want to build a model to predict which customers are most likely to churn.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector classifier(SVC), support vector machine(svm)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {'Age':[30,25,35,20,40,55,32,28],'MonthlyCharge': [50,60,80,40,100,120,70,55],'churn':[0,1,0,1,0,1,0,1]}
df = pd.DataFrame(data)
# print(df)
x = df.drop("churn",axis=1)
y = df['churn']
print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

print(x_train.shape)
print(x_test.shape)
print(x.shape)

svc_model = SVC(kernel = 'linear',C=1.0) #c is default regularization

svc_model.fit(x_train,y_train)

y_pred = svc_model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy: {accuracy}")

class_report = classification_report(y_test,y_pred)
print(f"class_report: {class_report}")

conf_mat = confusion_matrix(y_test,y_pred)
print(f"Confusion matrix: {conf_mat}")

# User's Input:
user_age = float(input("Please Enter Customer's age: "))
user_mon_charge = float(input("Please Enter Customer's monthly charge: "))

user_input = np.array([[user_age,user_mon_charge]])
prediction = svc_model.predict(user_input)
if prediction[0]==0:
    print("The customer is likely to stay")
else:
    print("The customer is at risk of churning")