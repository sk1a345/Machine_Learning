import pandas as pd
import numpy as np
import sklearn

df = pd.read_csv('synthetic_creditcard.csv')
# print(df)
print(df['Class'].value_counts())

#➡️➡️➡️Training model on imbalanced dataset:
from sklearn.model_selection import train_test_split
x = df.drop('Class',axis=1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=1)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)
# prediction on the test set:
y_pred = rf_model.predict(x_test)

#Calculate confustion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('confusion_matrix')
print(conf_matrix)

# Calculate classificant report:
class_report = classification_report(y_test,y_pred)
print("\nClassification Report: ")
print(class_report)
score  = rf_model.score(x_train,y_train)
print("Accuracy score: ",score)

#➡️➡️➡️Applying Sampling Techniques(RandomeOverSampler)
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# initialize RandomOverSampler
ros = RandomOverSampler()

# Perform Random Oversampling:

x_ros, y_ros = ros.fit_resample(x_train,y_train)

print(y_ros.value_counts())

# Initialize the model:
rf_model_ros = RandomForestClassifier()
# Training the model:
rf_model_ros.fit(x_ros,y_ros)
# Predicting on the test set:
y_pred_ros = rf_model_ros.predict(x_test)
# Calculating the confusion metrics:
co = confusion_matrix(y_test,y_pred_ros)
print("Confusion matrix(Random Oversampling): ")
print(co)
# Calculating the classification Report:
cro = classification_report(y_test,y_pred_ros)
print("Calssification report(Random Oversampler): ")
print(cro)

acc1 = rf_model_ros.score(x_test,y_test)
print("Accuracy Score: ",acc1)


#➡️➡️➡️Undersampling:
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
# Initialize RandomUnderSampler:

rus = RandomUnderSampler()
# Perform Random Undersampling:

x_rus, y_rus = rus.fit_resample(x_train,y_train)
print(y_rus.value_counts())

# Initialize the model:
rf_model_rus = RandomForestClassifier()
# Train the model on Random undersampled data
rf_model_rus.fit(x_rus,y_rus)
# predict on the test set:
y_pred_rus = rf_model_rus.predict(x_test)
# calculate the confusion metrix:
c = confusion_matrix(y_test,y_pred_rus)
print("Confusion matrix(random UnderSampling): ")
print(c)
# Calculate the classification report:
cr = classification_report(y_test,y_pred_rus)
print("\nCalssification report(RandomUndersampling): ")
print(cr)
accuracy1 = rf_model_rus.score(x_test,y_test)
print("accuracy1: ",accuracy1)
print()
#➡️➡️➡️SMOTE (Synthetically generate):
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report 
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x_train,y_train)
print(y_smote.value_counts())

#Initialize the model
rf_model_smote = RandomForestClassifier()
# Train the model on smote DATA
rf_model_smote.fit(x_smote,y_smote)
# predict on the test set:
y_pred_smote = rf_model_smote.predict(x_test)

# Calculate confusion matrix:
conf_matrix_smote = confusion_matrix(y_test,y_pred_smote) 
print("Confusion Matrix(SMOTE): ")
print(conf_matrix_smote)

# Calculating the classification_report

cr1 = classification_report(y_test,y_pred_smote)
print("Classification Matrix(SMOTE): ")
print(cr1)

# Calculate Accuracy:
accuracy_smote = rf_model_smote.score(x_test,y_test)
print("Accuracy(SMOTE): ",accuracy_smote)


#➡️➡️➡️Prediction:
# Get the input data as a 2d Array:
inputs_1 = x_test.iloc[[10]].values
# Predict with the mode:
prediction = rf_model_smote.predict(inputs_1)
print("Acutual class: ",y_test.iloc[111])
print("predicted Calss: ",prediction[0])
