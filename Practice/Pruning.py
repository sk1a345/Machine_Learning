import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\OneDrive\python_Pandas\Practice\Logistic_Regression.csv")
# print(df.head())

x = df.drop("Purchased",axis=1)
y = df['Purchased']

# print(df.isnull().sum())
#➡️➡️Filling the null values
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='most_frequent')

x = pd.DataFrame(si.fit_transform(x),columns=x.columns)
# print(x.isnull().sum())

#➡️➡️ Performing sampling:
# print(y.value_counts())
from imblearn.over_sampling import SMOTE
smote = SMOTE()

x,y = smote.fit_resample(x,y)
# print(y.value_counts())
# print(x.shape)
# print(y.shape)

#➡️➡️ Performing scaling on the input data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x = pd.DataFrame(sc.fit_transform(x),columns=x.columns)
# print(x) 
#➡️➡️Dividing data into training and testing data:
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# ➡️➡️Training the model using DecisionTreeClassifier:

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
# print(dt.score(x_train,y_train)) #0.9953560371517027
# print(dt.score(x_test,y_test))  #0.808641975308642

#➡️➡️ Applying prepruning(depth is decided earlier)

dt0 = DecisionTreeClassifier(max_depth=4)
dt0.fit(x_train,y_train)
print(dt0.score(x_train,y_train))
print(dt0.score(x_test,y_test))


#➡️➡️ Applying postpruning:
for i in range(1,20):
    dt1 = DecisionTreeClassifier(max_depth=i)
    dt1.fit(x_train,y_train)
    print(dt1.score(x_train,y_train)," i=",i)
    print(dt1.score(x_test,y_test)," i=",i,"\n")


#➡️➡️ Printing the decision tree region
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(10,8))
plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=dt)
plt.show()


# ➡️➡️  plotting tree:
plt.figure(figsize=(20,8))
from sklearn.tree import plot_tree
plot_tree(dt1, feature_names=x.columns, class_names=['Not Purchased','Purchased'], filled=True) #for dt0 only 4 levels will be printed of decision tree
plt.show()








