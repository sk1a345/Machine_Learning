import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer


df = pd.read_csv("SimpleImputer.csv")
print(df)

simple_mean = SimpleImputer(strategy = "mean")
simple_median = SimpleImputer(strategy = "median")
simple_mode = SimpleImputer(strategy = "most_frequent")
# df['A'] = simple_mean.fit_transform(df[['A']])
# df['B'] = simple_mode.fit_transform(df[['B']])
# df['C'] = simple_mean.fit_transform(df[['C']])
# df['D'] = simple_median.fit_transform(df[['D']])

# print(df)
# print(df.info())

df =df.fillna()
print(df)
