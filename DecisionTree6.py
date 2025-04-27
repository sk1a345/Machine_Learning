# You have been tasked with creating a decision tree model to predict whether a person is likely to purchase a new smartphone based on their age, income and education level. You are provided with a dataset containing these attributes and the target variable indiating whether the person made a purchase or not

# Step 1:
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2:
x = np.array([[25,50000,3],[35,90000,2],[40,60000,5],[45,80000,3],[20,30000,2],[55,120000,4],[28,40000,1],[32,100000,3],[38,75000,2]]) #age, income, education(1-12th, 2-graduate, 3-Masters, 4-researcher)

y = np.array([0,1,1,0,1,0,1,0,1])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

model =DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# UserInput:
age = float(input("Enter your age: "))
income = float(input("Enter your income: "))
education = float(input("Enter education level(1-12th, 2-graduate, 3-Masters, 4-researcher): "))

user_input = np.array([[age,income,education]])
prediction =model.predict(user_input)
if prediction[0]==1:
    print("User is likely to purchse a smart phone")
else:
    print("User is not Likely to purchase a smart phone")


#➡️➡️➡️IMP :This step trains the model using the historical data (age, income, education vs. purchase decision). Once the model is trained, it learns how to make predictions based on new data.You're using the already-trained model to make a prediction — you're not training the model again.Fitting is for learning from data (i.e., training).Predicting is for applying what the model has learned to new, unseen data. You would overwrite the model’s learning from the original dataset — which is not what you want. It would "forget" the previous data and learn from just one example, which isn't useful.