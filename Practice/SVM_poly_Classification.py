import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\polynomial_logistic_dataset.csv")
# print(df.head())

# printing the null values:
# print(df.isnull().sum()) #no null values present:

# Removing the duplicates:
# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape) #no duplicates present:

# visualizing the model:
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.scatterplot(x='data1',y='data2',data = df,hue='output')
plt.show()

# Dividing the data into the input and output data:

x = df.drop('output',axis=1)
y = df['output']

# Dividing the data into the training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# training the model:
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)

# Printing the accurary:
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("Classification Report: \n",classification_report(y_test,y_pred))
print("Accuracy score: ",accuracy_score(y_test,y_pred)*100)
print("Score(train): ",svc.score(x_train,y_train)*100)
print("Score(test): ",svc.score(x_test,y_test)*100)

# mlxtend:
plt.figure(figsize=(15,8))
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=svc)
plt.show()