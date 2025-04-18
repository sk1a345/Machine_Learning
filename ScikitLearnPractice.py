import pandas as pd
import numpy as np
import sklearn


'''df = pd.read_csv("SimpleImputer.csv")
print(df)
# print(df.head(4))
# Training and splitting the data 
x = df.drop('D',axis =1)
y = df['D']
# print(x)
# print(y)'''

# Train_test_split:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 1)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)

# Fit_transform:
# StandardScaler:

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler() 
# x_train_scaled = sc.fit_transform(x_train.head(5))
# x_test_scaled = sc.fit_transform(x_test)


#MinMaxScaler:

# from sklearn.preprocessing import MinMaxScaler

# sc = MinMaxScaler()
# x_train_scalled = sc.fit_transform(x_train)
# x_test_scalled = sc.transform(x_test)

# print(x_train_scalled)
# print(x_test_scalled)

'''from sklearn.impute import SimpleImputer

simple_mean = SimpleImputer(strategy = "mean")
df['A'] = simple_mean.fit_transform(df[['A']])

simple_median = SimpleImputer(strategy = "median")
df['B'] = simple_median.fit_transform(df[['B']])

simple_mode = SimpleImputer(strategy = "most_frequent")
df['C'] = simple_mode.fit_transform(df[['C']])

print(df)'''
