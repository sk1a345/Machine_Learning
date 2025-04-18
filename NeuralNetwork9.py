# Problem Statement:
# You are tasked with creating a neural network model to predict whether a student will pass or fial an exam based on two features:hours studied and previous exam scores. The dataset should contains information on hours studied and previous exam scores for a group of students, along with their exam outcomes(pass or fialt): 

# Step no 1:
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder #usually used for the yes/no type of data

# Step no. 2:
hours_studied = [2.5, 1.5, 3.0, 1.8, 4.0, 2.0, 3.5, 2.7, 0.5, 1.0, 1.2]
prev_exam_score = [80, 70, 75, 60, 85, 80, 90, 65, 40, 50, 55]
exam_outcome = ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Fail', 'Fail', 'Fail']

# Step no. 3:

label_encoder = LabelEncoder()
encoded_exam_outcome =label_encoder.fit_transform(exam_outcome)
print(encoded_exam_outcome) #pass =1, fail =0

# Step no. 4:

df = pd.DataFrame({'Hours_study':hours_studied,"PreviousExamScore":prev_exam_score,"Exam_Result":encoded_exam_outcome})
# print(df)

x = df.drop('Exam_Result',axis=1)
y = df['Exam_Result']
# print(x)
# print(y)

# Step no. 5:

clf = MLPClassifier(hidden_layer_sizes =(4,),activation = 'logistic',max_iter=1000,random_state =1)
clf.fit(x,y)

# Step no. 6:(input of some example data from user: )

hr_study = float(input("Enter the hours studied: "))
pre_exam_score = int(input("Enter the previous exam score: "))

user_input = pd.DataFrame({"Hours_study":[hr_study],"PreviousExamScore":[pre_exam_score]})

# Step no. 7:
predicted_outcome = clf.predict(user_input)

# Step no. 8:

predicted_outcome_decode = label_encoder.inverse_transform(predicted_outcome)

# Step no. 9:
print(f"Predicted Exam outcome for the new student: {predicted_outcome_decode[0]}")


