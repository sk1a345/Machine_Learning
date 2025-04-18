# Mathematics Transformers:
# Function Transformer(use only when data is not normally distributed)

# Log T
# Reciprocal T
# Sq/ Sq root 

import pandas as pd
import numpy as np
import sklearn

df = pd.read_csv('FunTranss.csv')
df = df[['Survived','Age','Fare']]
# print(df.isnull().sum())
# print(df)

from sklearn.impute import SimpleImputer
i = SimpleImputer(strategy='mean')

df['Age'] = i.fit_transform(df[['Age']])
# print(df.to_string())
# mean = df['Age'].mean()
# print(mean)

df['Fare'].fillna(df['Fare'].mean(),inplace=True)
print(df['Fare'].mean())
# print(df['Fare'])

x = df.iloc[:,1:3]
y = df.iloc[:,0]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state=1)


# Training models before function transformer:

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf.fit(x_train,y_train)
clf2.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_pred1 = clf2.predict(x_test)

print("\nNormal: ")
print("Accuracy LR : ",accuracy_score(y_pred,y_test))
print("Accuracy DT: ",accuracy_score(y_pred1,y_test))

# LogTransformer
trf = FunctionTransformer(func = np.log1p)
x_train_transformed = trf.fit_transform(x_train)
x_test_transformed = trf.transform(x_test)

clf.fit(x_train_transformed,y_train)
clf2.fit(x_train_transformed,y_train)
y_pred = clf.predict(x_test_transformed)
y_pred1 = clf2.predict(x_test_transformed)

print("\nLogTransform: ")
print("Accuracy Lr: ",accuracy_score(y_test,y_pred))
print("Accuracy DT: ",accuracy_score(y_test,y_pred1))

# Reciprocal_transformer:

def reciprocal_tran(X):
    X = X+1e-6
    return 1/X
# Create the FunctionTransformer:
trf = FunctionTransformer(func = reciprocal_tran)

x_train_trans = trf.fit_transform(x_train)
x_test_trans = trf.transform(x_test)

# fitting classifiers
clf.fit(x_train_trans,y_train)
clf2.fit(x_train_trans,y_train)

y_pred = clf.predict(x_test)
y_pred1 = clf.predict(x_test)

print("\nReciprocal: ")
print("Accuracy LR: ",accuracy_score(y_test,y_pred))
print("Accuracy DT: ",accuracy_score(y_test,y_pred1))

# Square-root Transformer
def square(X):
    return X**2

sq = FunctionTransformer(func=square)

x_train_trans = sq.fit_transform(x_train)
x_test_trans = sq.transform(x_test)

clf.fit(x_train_trans,y_train)
clf2.fit(x_train_trans,y_train)

y_pred = clf.predict(x_test)
y_pred1 = clf2.predict(x_test)

print("\nSquare")
print("Accuracy LR: ",accuracy_score(y_test,y_pred))
print("Accuracy DT: ",accuracy_score(y_test,y_pred1))


# Squareroot
def sq_root(X):
    return np.sqrt(X)

# print(sq_root(25)) 
sr = FunctionTransformer(func=sq_root)

x_train_trans = sr.fit_transform(x_train)
x_test_trans = sr.transform(x_test)

clf.fit(x_train_trans,y_train)
clf2.fit(x_train_trans,y_train)

y_pred = clf.predict(x_test)
y_pred1 = clf2.predict(x_test)

print("\nSquare Root")
print("Accuracy LR: ",accuracy_score(y_pred,y_test))
print("Accuracy Dt: ",accuracy_score(y_pred1,y_test))


