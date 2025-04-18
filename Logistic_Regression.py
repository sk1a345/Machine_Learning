import pandas as pd
import numpy as np
import sklearn


# url = "https://raw.githubusercontent.com/611noorsaeed/100-days-Scikit-Learn-Tutorials-/refs/heads/main/7%20churn.csv"
# df = pd.read_csv(url)
# print(df) 

df = pd.read_csv("churn.csv")
# print(df)

columns_to_keep = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','Contract','TotalCharges','Churn']

df = df[columns_to_keep]
# print(df.head())

# It is the superwised model (since the output column is given (classification algo ->Logistic regression))

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# df['gender'] = label_encoder.fit_transform(df['gender'])

#➡️➡️➡️➡️Encode Binary varibles(e.g., Yes/No columns):
 
categorical_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','Contract',"Churn"]
for cols in categorical_cols:
    df[cols] = label_encoder.fit_transform(df[cols])

# print(df.head())
# ->To check
# for i in categorical_cols:
#     print(df[i].value_counts())
'''
#➡️➡️➡️➡️Temporary solutions(Enoding the binary varibales:)

binary_columns = ['Partner','Dependents','PhoneService',"Churn"]

df[binary_columns] = df[binary_columns].replace({"Yes":1,"No":0})

# print(df['MultipleLines'].value_counts())

df['gender'] = df['gender'].replace({"Female":0,"Male":1})
print(df)'''

#➡️➡️➡️➡️Split the DAtaset into training and testing sets:

x = df.drop("Churn",axis =1)
y = df["Churn"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#➡️ Convert "TotalCharges" column to float, and handle errors = "coerce" to replace non0numeric value with NaN
#➡️print(x_train.info())

x_train['TotalCharges'] = pd.to_numeric(x_train['TotalCharges'],errors = "coerce")
x_test['TotalCharges'] = pd.to_numeric(x_test['TotalCharges'],errors = "coerce")

#➡️Replace missing values in the 'Totalcharges column with the mean of the column:
x_train['TotalCharges']=x_train['TotalCharges'].fillna(x_train['TotalCharges'].mean())
x_test['TotalCharges']=x_test['TotalCharges'].fillna(x_test['TotalCharges'].mean())


#➡️➡️➡️➡️➡️Standardize features(optional but often beneficial for logistic regression)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #will ne converted into the numpy array:

#➡️➡️➡️➡️➡️Logistic regression:

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(x_train,y_train)
y_predi = lg.predict(x_test)

# print(y_predi)

# ➡️mean_squared_error

'''from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_predi)
print(mse)'''

#➡️another way to check accuracy
from sklearn.metrics import accuracy_score
a = accuracy_score(y_test,y_predi)
# print(a)

#➡️➡️➡️➡️➡️Save Model
import pickle
pickle.dump(lg,open('logistic_model','wb')) #write binary


# ➡️➡️Classification system:

def prediction(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge):
    data = {
    'gender' : [gender],
    'SeniorCitizen' : [Seniorcitizen],
    'Partner' : [Partner],
    'Dependents' : [Dependents],
    'tenure' : [tenure], 
    'PhoneService' : [Phoneservice],
    'MultipleLines' : [multiline],
    'Contract' : [contact],
    'TotalCharges' : [totalcharge]
    }
    df1 = pd.DataFrame(data)
    # Encoding the Categorical columns:

    categorical_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','Contract']    
    for cols in categorical_cols:
        df1[cols] = label_encoder.fit_transform(df1[cols])
    
    df1 = scaler.transform(df1)
    result = lg.predict(df1).reshape(1,-1)
    return result[0]

gender = "Female"
Seniorcitizen = "No"
Partner = "Yes"
Dependents = "No"
tenure = 1
Phoneservice = "No"
multiline = "No"
contact = "Month-to-month"
totalcharge = 29.85

result = prediction(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge)

if result==0:
    print("Not Churn")
else:
    print("Churn")


