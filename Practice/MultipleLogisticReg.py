import pandas as pd
import numpy as np

#➡️ Loading the dataset:

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\balanced_overlap_placement.csv")

# Visualizing the data:
import seaborn as sns
import matplotlib.pyplot as plt
'''
sns.scatterplot(x='CGPA',y ='Score',hue='Placed',data = df)
plt.title("Data Visualization")
plt.xlabel("CGPA")
plt.ylabel("Score")
plt.show()'''

# print(df.head())

#➡️ checking for the null values
# print(df.isnull().sum()) 

#➡️Dividing data into input and output data :
x = df.drop('Placed',axis=1)
y = df['Placed']

# print(x)
# print(y)

#➡️ Performing scaling on the input features:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x),columns=['CGPA','Score']) #since fit_transform converts data into numpy array:
# print(x)

# print(y.value_counts())

#➡️➡️ Dividing data into training and testing data:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=1)


#➡️ Performing the Sampling on the entire dataset:
# ➡️➡️➡️➡️No need to perform sampling on this dataset:
# from imblearn.over_sampling import SMOTE
# smote = SMOTE()
# x_resampled,y_resampled = smote.fit_resample(x_train,y_train)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(x_resampled.shape)
# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y_resampled.shape)

# # counting values:
# print(y_train.value_counts())
# print(y_resampled.value_counts())

#➡️➡️ Training the model:
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()

lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)

#➡️➡️Checking efficiency:
from sklearn.metrics import confusion_matrix,classification_report

con_mat = confusion_matrix(y_test,y_pred)
print("Confusion matrix: \n",con_mat)

class_re = classification_report(y_test,y_pred)
print("Classification report: \n",class_re)

# ➡️➡️checking score:
print("Score: ",lg.score(x_test,y_test))

from mlxtend.plotting import plot_decision_regions
plt.figure(figsize=(9,6))
plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=lg)
# plt.show()

# taking the user input:
user_cgpa = float(input("Enter your cgpa: "))
user_score = float(input("Enter your score: "))


# But your scaler sc was trained on columns named "CGPA" and "Score" — not "user_cgpa" a
user_input = pd.DataFrame([[user_cgpa,user_score]],columns=["CGPA",'Score'])
print(user_input)
# scaling the user_input:
user_input_scaled = pd.DataFrame(sc.transform(user_input),columns=["CGPA",'Score'])
print(user_input_scaled)

user_placed_predict = lg.predict(user_input_scaled)
print("Predicted user placement: ",user_placed_predict)

