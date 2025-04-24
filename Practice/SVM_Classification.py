import pandas as pd
import numpy as np

#➡️➡️ loading the data:

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\balanced_overlap_placement.csv")
# print(df.head())

#➡️➡️ Finding the null values the data:
# print(df.isnull().sum()) #no null values present:

#➡️➡️ Removing the duplicates:
# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape)

#➡️➡️ visualizing the data:

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='CGPA',y='Score',data = df,hue='Placed')
# plt.show()


#➡️➡️ Diving the data into input and output data:
x = df.drop('Placed',axis=1)
y = df['Placed']

#➡️➡️ Performing the Sampling on the input data:

# print(y.value_counts()) #no need of performing sampling data is already balanced:

#➡️➡️ performing scaling on the input data:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x),columns=x.columns)
# print(x)

# Dividing the data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=14)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# Training the model:
from sklearn.svm import SVC
svc = SVC(kernel="linear")

svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)

# Testing the accuracy of the model:
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print("Accuracy score: ",accuracy_score(y_test,y_pred)*100)
print("Classification report: \n",classification_report(y_test,y_pred))
print("Confusion metrics: \n",confusion_matrix(y_test,y_pred))
print("Score(train): ",svc.score(x_train,y_train)*100)
print("Score(test): ",svc.score(x_test,y_test)*100)

# using the data using the decision_plot:
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=svc)
plt.show()

# Visualizing the data:

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

