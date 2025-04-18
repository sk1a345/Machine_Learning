# In an e-commerce company,the management wants to pedict whether a customer will purchase a high-value product based on their age,time spent on the website and whether they have added items to their cart. The goal is to optimize marketing strategies by targeting potential customers more effectively, thereby increasing sales and revenue

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x = np.array([[25,30,0],[30,40,1],[20,35,0],[35,45,1]])
print(x)

y = np.array([0,1,0,1])

print(y)

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=1)

model = LogisticRegression()

model.fit(x_train,y_train,)

accuracy = model.score(x_test,y_test)
print(f"Model Accuracy: {accuracy}")

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error,confusion_matrix,classification_report


acc = mean_squared_error(y_test,y_pred)
print(f"mean_squared_error: {acc}")

confu_mat = confusion_matrix(y_test,y_pred)
print(f"Confusion metrix: {confu_mat}")

class_report = classification_report(y_test,y_pred)
print(f"Classification Report: {class_report}")

user_age = float(input("Enter Customer age: "))
user_time_spent = float(input("Enter time spent on website: "))
user_added_to_cart = int(input("Enter 1 if added to cart, else enter 0: "))

user_data = np.array([[user_age,user_time_spent,user_added_to_cart]])
prediction = model.predict(user_data)
if prediction[0] ==1:
    print("the customer is likely to purchase")
else:
    print("The customer is unlikely to purchase")
