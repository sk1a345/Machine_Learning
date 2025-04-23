import pandas as pd
import numpy as np

df = pd.read_csv("Practice/Logistic_Regression.csv")
# print(df.head())

# null values check: 
# print("Printing the null values: \n",df.isnull().sum())

# Filling the null values with mode:
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(si.fit_transform(df),columns=df.columns)
# print(df.isnull().sum())


# dividing the data into input and output data;
x =df.drop("Purchased",axis=1)
y = df['Purchased']


# Sampling the input data:

# print(df['Purchased'].value_counts())
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=1)
x_resampled ,y_resampled = smote.fit_resample(x,y)
# print(y_resampled.value_counts())


# Scaling the input data:
from sklearn.preprocessing import StandardScaler
si = StandardScaler()

x_resampled = pd.DataFrame(si.fit_transform(x_resampled),columns=x.columns)
# print(x_resampled)
# print(type(x_resampled))

# splitting the data into training and testing data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=1)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# Visualizing the data using the  
import matplotlib.pyplot as plt
import seaborn as sns

# sns.scatterplot(x="Age",y='EstimatedSalary',data=df,hue='Purchased')
# plt.show()

# Training the model:
from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier(n_neighbors=27)

knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

# Checking the accuracy:
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Confusion metrics: \n",confusion_matrix(y_test,y_pred))
print("Classification report: \n",classification_report(y_test,y_pred))
print("Accuracy score: ",accuracy_score(y_test,y_pred)*100,"%")
print("Score(train): ",knn.score(x_train,y_train)*100,"%")
print("Score(test): ",knn.score(x_test,y_test)*100,"%")


# Checking for the overfitting of the model:
'''for i in range(20,30):
    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn1.fit(x_train,y_train)
    y_pred1 = knn1.predict(x_test)
    # print("Score(train): ",knn1.score(x_train,y_train)*100," Socre(test): ",knn1.score(x_test,y_test)*100, i,'\n')
'''

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



