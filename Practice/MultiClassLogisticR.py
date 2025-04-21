import pandas as pd
import numpy as np

df = pd.read_csv('Practice/iris_dataset.csv')
# print(df)

print(df['species'].unique()) #(0 = Setosa, 1 = Versicolor, 2 = Virginica)

# Plotting the graph in the form of pairplot:
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data = df,hue = 'species')
# plt.show()

x = df.drop('species',axis=1)
y = df['species']

# Performing the scaling on the data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x),columns=['sepal length','sepal width','petal length','petal width'])
print(x)

# Splitting the data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# Training the model:
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)

# Classification report:
from sklearn.metrics import classification_report
class_re = classification_report(y_test,y_pred)
print("Classification Report:\n",class_re)

# Confusion metrics:
from sklearn.metrics import confusion_matrix
con_me = confusion_matrix(y_test,y_pred)
print("Confusion metrics: \n",con_me)

# Printing score:
print("Score: ",lg.score(x_test,y_test)*100," %")

# using the ovr method 
lg1 = LogisticRegression(multi_class= 'ovr')
lg1.fit(x_train,y_train)
y_pred1 = lg1.predict(x_test)

# Classification report:
print("Classification report: \n",classification_report(y_test,y_pred1))
# Confusion metrics:
print("Confusion metrics: \n",confusion_matrix(y_test,y_pred1))
# Score:
print("Score: ",lg1.score(x_test,y_test)*100," %")

# using the multinomial method:
lg2 = LogisticRegression(multi_class='multinomial')
lg2.fit(x_train,y_train)
y_pred2 = lg2.predict(x_test)

# Classification report:
print("Classification report: \n",classification_report(y_test,y_pred2))
# confusion metrics:
print("Confusion metrics: \n",confusion_matrix(y_test,y_pred2))
# Score:
print("Score: ",lg2.score(x_test,y_test)*100," %")

