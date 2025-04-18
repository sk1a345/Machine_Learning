# Problem -statement:use a Random Forest Classifier to predict whether a person is likely to purchase a product based on certain features like age, gender and estimated salary:

# Step 1:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



# Step 1:
x = np.array([[10,1,10000],[20,0,5467],[32,0,7000],[40,1,8000],[18,0,5000],[24,1,2000],[26,1,9000],[56,0,7000], [46,1,8750],[78,0,9000],[55,1,5555]]) #male =1,female = 0,(age,gender,salary)

y = np.array([1,0,0,1,0,0,1,0,1,1,0])

# Step 2:
# Train-test-split:

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# Step 3:
scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

# step 4:

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)

acc = accuracy_score(y_test,y_pred)
# print(acc)

# Step 5:
# User Input:
age = float(input("Enter age: "))
gender =float(input("Enter gender(0-female,1-male): "))
salary = float(input("Enter the salary: "))

user_input = np.array([[age,gender,salary]])
user_input_scaled = scaler.transform(user_input)
predict = rfc.predict(user_input_scaled)

if predict[0] ==1:
    print("Person is likely to purchase")
else:
    print("person is not likely to purchase")





