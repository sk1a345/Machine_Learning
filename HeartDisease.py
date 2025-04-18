import pandas as pd
import numpy as np
import sklearn

df = pd.read_csv("HeartDisease.csv")
# print(df.shape)
# print(df)

# dropping the unimportant colum

df.drop("education",axis =1,inplace = True)
# print(df.head())
# print(df.isnull().sum())
# Defining the binary columns:
bin_cols = ['male','currentSmoker','prevalentStroke','prevalentHyp','diabetes']

# Fill missing values for binary features with the most frequent value(mode)
for col in bin_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val,inplace = True)

numeric_cols = ['cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose']
for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val,inplace=True)

# print(df.isnull().sum())
# Balancing the Dataset:
print(df["TenYearCHD"].value_counts())

from sklearn.utils import resample
# Seperate the majority and minority classes:
df_majority = df[df['TenYearCHD']==0]
df_minority = df[df['TenYearCHD'] ==1]

# upsample minority class:
df_minority_upsampled = resample(df_minority,
                                replace=True,#sample with replacement
                                n_samples=len(df_majority),#to match majority
                                random_state = 1)#Reproducible results:

# Combine majority class with upsampled minority class

df_balanced = pd.concat([df_majority,df_minority_upsampled])
print(df_balanced['TenYearCHD'].value_counts())

# Train_Test_Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# separate features (X) and target variable (y)
x = df_balanced.drop(columns = ['TenYearCHD'])
y = df_balanced['TenYearCHD']

# Split the data into training and testing sets(80% - train, 20% - test)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# Scaling(StandardScaler):

# Initializing the standardScaler:
scaler = StandardScaler()
#Fit teh scaler to training data and transform both training and testing data

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Instantiate the RandomForestClassifier:

# rf_classifier = RandomForestClassifier()
# # Train the RandomForestClassifier:
# rf_classifier.fit(x_train_scaled,y_train)
# # Predict on the test set:
# y_pred_rf = rf_classifier.predict(x_test_scaled)
# # Calculate the accuracy:
# accuracy_rf = accuracy_score(y_test,y_pred_rf)
# print("Random Forest Classifier Accuracy: ",accuracy_rf)
# # classification report:
# print("Classification Report for Random Forest Classifier: ")
# print(classification_report(y_test,y_pred_rf))
# # Confusion Matrix:
# print("Confusion Matrix for Random Forest Classifier: ")
# print(confusion_matrix(y_test,y_pred_rf))

# Training 10 Models with different Metrics:

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Define a List of Classifiers:
classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
]

# Create a dictionary to store the results:
results = {}
# Train and evaluate each classifier

for clf in classifiers:
    clf_name = clf.__class__.__name__
    clf.fit(x_train_scaled,y_train)
    y_pred = clf.predict(x_test_scaled)

    # Calculate accuracy:
    accuracy = accuracy_score(y_test,y_pred)
    # print(f"{clf_name} Accuracy : {accuracy}")

    # Classification Report
    # print(f"Classification Report for {clf_name}: ")
    # print(classification_report(y_test,y_pred))

    # Confusion matrix:
    # print(f"Confusion Matrix for {clf_name}: ")
    # print(confusion_matrix(y_test,y_pred))
    # print("="*50)



# Best Model(Random Forest Classifier):
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

rf_classifier= RandomForestClassifier()

rf_classifier.fit(x_train_scaled,y_train)

y_pred_rf = rf_classifier.predict(x_test_scaled)

accu_rf = accuracy_score(y_test,y_pred_rf)
print("Random Forest Classifier Accuracy: ",accu_rf)

print("Classification Report for Random Forest Classifier: ")
print(classification_report(y_test,y_pred_rf))

print("Confusion Matrix for Random Forest Classifier: ")
print(confusion_matrix(y_test,y_pred_rf))

# Testing the data:
# test1
print("predicted class: ",rf_classifier.predict(x_test_scaled[100].reshape(1,-1))[0])
print("Actual class: ",y_test.iloc[100])

# for i in range(1,10):
#     print("predicted class: ",rf.predict(x_test_scaled[i].reshape(1,-1))[0])
#     print("Actual class: ",y_test.iloc[i])


import pickle
pickle.dump(rf_classifier,open("rf_classifier.pkl",'wb'))
pickle.dump(scaler,open("scaler.pkl",'wb'))

# Load Models to test1:
import pickle
# Load the RandomForestClassifier model

import pickle

# Load the RandomForestClassifier model
with open("rf_classifier.pkl", "rb") as file:
    rf_classifier = pickle.load(file)

# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Predictive System:
import numpy as np
def predict(model,scaler,male,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose):
    # Encode categorical variables:
    male_encoded = 1 if male.lower() =='male' else 0
    currentSmoker_encoded =1 if currentSmoker.lower()=='yes' else 0
    BPMeds_encoded = 1 if BPMeds.lower()=='yes' else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() =='yes' else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() =='yes' else 0
    diabetes_encoded = 1 if diabetes.lower()=='yes' else 0

    #prepare features array:
    features = np.array([[male_encoded,age,currentSmoker_encoded,cigsPerDay,BPMeds_encoded,prevalentStroke_encoded,prevalentHyp_encoded,diabetes_encoded,totChol,sysBP,diaBP,BMI,heartRate,glucose]])

    #Scalling
    scaled_features = scaler.transform(features)
    
    # predict by model:
    result = model.predict(scaled_features)
    return result[0]

male = "female"
age = 56.00
currentSmoker = "yes"
cigsPerDay = 3.00
BPmeds = "no"
prevalentStroke = "no"
prevalentHyp = "yes"
diabetes = "no"
totChol = 285.00
sysBP = 145.00
diaBP =100.00
BMI = 30.14
heartRate = 80.00
glucose = 86.00

result = predict(rf_classifier,scaler,male,age,currentSmoker,cigsPerDay,BPmeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose)

if result==1:
    print("The Patient has Heart Disease")
else:
    print("The Patient has No Heart Disease")
