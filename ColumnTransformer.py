import pandas as pd
import numpy as np
import sklearn

url = "https://raw.githubusercontent.com/611noorsaeed/100-days-Scikit-Learn-Tutorials-/refs/heads/main/9%20covid_toy.csv"

# df = pd.read_csv(url)
df = pd.read_csv('Column_Transform.csv')
# print(df)
# print(df.isnull().sum())
# print(df.shape)

# print(df['city'].value_counts())

from sklearn.model_selection import train_test_split
x = df.drop('has_covid',axis =1)
y = df['has_covid']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# print(x.shape)
# print(x_train.shape)
# print(y.shape)
# print(y_test.shape)

#Filling the missing values:

from sklearn.impute import SimpleImputer
si = SimpleImputer()
x_train_fever = si.fit_transform(x_train[['fever']])
x_test_fever = si.fit_transform(x_test[['fever']])
d = pd.DataFrame(x_train_fever)
# print(d.isnull().sum())

# Encoding categorical values:
from sklearn.preprocessing import OrdinalEncoder
Ordinal = OrdinalEncoder()
x_train_cough = Ordinal.fit_transform(x_train[['cough']])
x_test_cough = Ordinal.fit_transform(x_test[['cough']])
# print(x_train_cough)

# gender and city
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop = 'first',sparse_output =False)

x_train_gender_city = ohe.fit_transform(x_train[['gender','city']])
x_test_gender_city = ohe.transform(x_test[['gender','city']])
# print(x_train_gender_city.shape)
# print(x_train_gender_city)

x_train_age = x_train.drop(columns = ['gender','fever','cough','city']).values
x_test_age = x_test.drop(columns = ['gender','fever','cough','city']).values

# print(x_train_age)

# x_train_transformed = np.concatenate((x_train_age,x_train_fever,x_train_gender_city,x_train_cough),axis=1)
# print(x_train_transformed) 

# Column Transformer:

from sklearn.compose import ColumnTransformer

tf = ColumnTransformer(
transformers=[
    ("tf1",SimpleImputer(),['fever']),
    # ("tf2",OrdinalEncoder(),['cough']),
    ('tf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tf3',OneHotEncoder(drop='first',sparse_output = False),['gender','city']),
],
remainder = "passthrough"
)
# print(tf)
x_train_tf = tf.fit_transform(x_train)
x_test_tf = tf.fit_transform(x_test)
print(x_train_tf)

