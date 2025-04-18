# A retuail company wants to predict customer purchasing behavior based on their age, salary and past purchase history. The company aims to use k-nearest neighbors (KNN) algo to classify customers into potential buying groups to personalize marketing strategies. This predictive model will help the company understand and target specific customer segments more effectively, thereby increasing sales and customer satisfaction.


# Step 1:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# Step 2:

data = np.array([[25,50000,2],[30,80000,1],[35,60000,3],[20,30000,2],[40,90000,1],[45,75000,1]])
labels = np.array([1,2,1,0,2,1]) #0:low,1:medium,2:high

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size =0.2,random_state=1)


# print(x_train)
# Step no. 
# print(data.shape)
# print(x_train.shape)
# print(x_test.shape)

# print(labels.shape)
# print(y_train.shape)
# print(y_test.shape)
# Step no. 4:

scaler = StandardScaler()
# x_train =scaler.fit_transform(data)
# y_train = labels

# Step no. 5:
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

# accuracy = knn.score(x_test,y_test)
# print(accuracy)

# User input:
user_input = np.array([[30,80000,1]])
user_input_scaled = scaler.transform(user_input)

output =knn.predict(user_input_scaled)

print(f"Customer behavior is: {output[0]}")




