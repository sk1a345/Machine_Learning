import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\polynomial_logistic_dataset.csv")
# print(df.head())

# Visualizing the data:

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='data1',y='data2',data = df,hue='output')
# plt.show()

# Seperating the data into input and output data:
x = df.drop('output',axis=1)
y = df['output']

# Applying the polynomial features on the data:
from sklearn.preprocessing import PolynomialFeatures
pl = PolynomialFeatures(degree=6) #increasing the degree leads the model to the overfitting
pl.fit(x)
x = pd.DataFrame(pl.transform(x))
print(x)


# Splitting data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# print(y.value_counts())

# Training the model:
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

# Classification report and confusion metrics:
from sklearn.metrics import classification_report,confusion_matrix

class_re = classification_report(y_test,y_pred)
print("classification report is:\n ",class_re)

conf_me = confusion_matrix(y_test,y_pred)
print("Confusion Metrics is: \n",conf_me)

# printing score:
print("Score: ",lr.score(x_test,y_test)*100," %")

# plotting the graph using the mlextend:

# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf = lr)
# # plt.show()

