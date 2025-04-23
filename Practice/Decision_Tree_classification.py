import pandas as pd
import numpy as np

df = pd.read_csv("Practice/Logistic_Regression.csv")
# print(df.head())

# Plotting the graph:
import matplotlib.pyplot as plt
import seaborn as sns

# sns.scatterplot(x='Age',y='EstimatedSalary',hue='Purchased',data=df)
# plt.show()


# Splitting the data into input  and output data:
x= df.drop('Purchased',axis=1)
y = df['Purchased']

# Filling the missing values using the simple imputer:

# print(x.isnull().sum())
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='most_frequent')
ar = si.fit_transform(df[['Age','EstimatedSalary']])
x = pd.DataFrame(ar,columns=x.columns)
# print(x.isnull().sum())

# print(x)
# print(y)

# Scalling the input features:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x),columns=x.columns)
# print(x)

# Splitting the data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# print(x.shape)
# print(x_test.shape)
# print(x_train.shape)


# sampling the data:
# print(y.value_counts())
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=1)
x_resampled, y_resampled = smote.fit_resample(x_train,y_train)
print(x_resampled.shape)
print(y_resampled.shape)
print(x_test.shape)

#Training the model using the Decision tree classification model:
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_resampled,y_resampled)

y_pred = dt.predict(x_test)
# Printing score:
print("Score: ",dt.score(x_test,y_test)*100," %")

# Confusion metrics,classification_report,accuracy_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print("Confusion metrics: \n",confusion_matrix(y_test,y_pred))
print("Classification report : \n",classification_report(y_test,y_pred))
print("Accuracy: ",accuracy_score(y_test,y_pred)*100," %")


# User input:

user_age = float(input("Enter your age: "))
user_sal = float(input("Enter your salary: "))

user_input = pd.DataFrame([[user_age,user_sal]],columns=x.columns)
user_input = pd.DataFrame(sc.transform(user_input),columns=x.columns)


# predicting the output:
pred_user_purchased = dt.predict(user_input)

# print(pred_user_purchased[0])
if pred_user_purchased[0]==1:
    print("User is going to purchase(1)")
else:
    print("User is not going to purchase(0)")


# Plotting the decision tree:

from sklearn.tree import plot_tree

plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=x.columns, class_names=['Not Purchased','Purchased'], filled=True)
plt.show()


