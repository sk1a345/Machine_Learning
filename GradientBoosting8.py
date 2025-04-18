# Create a predictive model suing Gradient Boosting to forecast housing prices based on various features such as square footage,number of bedrooms,number of bathrooms, and location

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

# Step no. 2:
data = {'squareFootage': [1500,2000,1200,1800,1350],
        'Bedrooms': [3,4,2,3,3],
        'Bathrooms': [2,2.5,1.5,2,2],
        'Location':['Suburb','City','Rular','City','Suburb'],
        'Price': [250000,300000,180000,280000,220000]
}

df = pd.DataFrame(data)
# print(df.head())

# # Converting the location column to dummy:
# df = pd.get_dummies(df,columns=['Location'])
# print(df.head)

df['Location'] = df['Location'].replace({'Suburb':0,'City':1,'Rular':2})
# print(df)

x = df.drop('Price',axis=1)
y = df['Price']
# print(x)
# print(y)

# Train_Test_split:

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

model = GradientBoostingClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
# print(mse)

# User input:
print("Enter the details for the house prediction: ")
sq_footage = float(input("Square Footage: "))
bedrooms = int(input("Enter no. of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))
location = input("enter Location (Suburb/City/Rular): ")

if location=='Suburb':
    location =0
elif location== 'City':
    location =1
else:
    location =2

# print(location)

user_input = pd.DataFrame({
    'squareFootage':[sq_footage],
    'Bedrooms':[bedrooms],
    'Bathrooms':[bathrooms],
    'Location':[location]
})
print(user_input)

# Predication of model on user Input:
predicted_price = model.predict(user_input)
print(f"Predicted price for the house is: {predicted_price[0]}")





