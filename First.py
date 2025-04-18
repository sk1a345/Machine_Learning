import numpy as np
import pandas as pd

dict1 = {
    "name":["harry","Sneha","Kari","Kiran","Samisksha"],
    "marks":[1,100,200,50,50],
    "city":["Mumbai","Nagpur","pune","Chennai","Bhandara"]
}
df = pd.DataFrame(dict1)

df.to_csv("friends.csv") 
# print(df)
'''
df = pd.read_csv("friends.csv")
print(df)
'''
'''import os
if os.path.exists("friends.csv"):
    print("file exists") 
else:
    print("File not found")'''
# print(df.head(2))
'''newfile = pd.read_csv("friends.csv")
print(newfile)
df.index = ['first','second','third','fourth','fifth']
print(df)
print("\n",newfile)
'''
# Pandas has two types of data structures:
'''
a)Series- It's a one dimensional array with indexes,it stores a single column or row of the data in a dataFrame
b)Dataframe - It's a tabular spreadsheet like structure representing rows each of which contains one or multiple columns:
=> A one-dimentsional array(labeled) capable of holding any type of dta -Series
=> A tow dimensional data(labeled) structure with columns of potentially different types of data -Dataframe
'''
'''# Series:
ser = pd.Series(np.random.rand(10))
print(ser)
print(type(ser))

rng = np.random.default_rng()
print(rng.random(4)) #floating point numbers:
# print(pd.Series(rng.integers(low=2,high=10,size=(1,5))))
'''
# DataFrame:
newdf = pd.DataFrame(np.random.rand(334,5),index = np.arange(334))
'''# print(newdf)
# print(newdf.head())
# print(type(newdf))
print(newdf.describe())
print(newdf.shape)
print(newdf.size)
# newdf[0][0] = "Sneha"
print(newdf.dtypes)
newdf[0][0] = "Sneha"
print(newdf.head())
print(newdf.dtypes)
print(newdf.index)
print(newdf.columns)
print(newdf.to_numpy())'''
'''# Sorting the data(ascending order =>bydefault)
descending_order = newdf.sort_index(axis=0,ascending=False) #Row-wise
print(descending_order)
descending_order[0][0] = 34.5 # It creates the copy 
print(descending_order)
# print(newdf)

#column-wise:
column_des = newdf.sort_index(axis=1,ascending=False)
print(column_des)
column_des[0][0] = 78.345
print(column_des)
'''

# print(newdf[0])
# print(type(newdf[0]))
# print(newdf[0].dtype)
#copy and view concept:
'''newdf1 = newdf
print(newdf)
newdf1[0][0] = "sneha"
print(newdf1)
print(newdf) # here in this whole commented segment view will be created not the copy'''
'''newdf2 = newdf.copy()
print(newdf)
newdf2[0][0] = 56 #setting with copy warning will be shown
print(newdf2)
print(newdf) '''

# To remove the SettingWithCopyWarning:
'''newdf3 = newdf.copy()
newdf3.loc[0,0] = "Sneha"
print(newdf3)
print(newdf)'''
'''
#.loc[]
print(newdf.columns)
newdf.columns = list("ABCDE")
print(newdf.head(4))
newdf.loc[0,'C'] = "kari"
print(newdf)
newdf.loc[0,0] = "Sneha"
print(newdf)
newdf =newdf.drop(0,axis = 1) #mentioning the axis is important otherwise it will remove it from the row(bydefault axis = 1)
print(newdf)
print((newdf.loc[[2,3],['A','B']]))
print("size = ",(newdf.loc[[2,3],['A','B']]).size)
print(newdf.loc[[1,2],:])
print(newdf.loc[:,['A','B']])
print(newdf.loc[(newdf['A']<0.4)& newdf['C']>0.3])'''
'''# difference between the loc and iloc 
# loc => count with the reference of name
# iloc =>count with the reference of index
print(newdf)
newdf.columns = list("ABCDE")
print(newdf.iloc[0,2])
print(newdf)
print(newdf.iloc[[0,5],[1,2]])'''

# Dropping the row and column
newdf.columns = list("ABCDE")
# print(newdf.drop(0)) #bydefault row will be dropped it is showing the copy
print(newdf.drop('A',axis=1))
newdf1 = newdf.drop(0)
print(newdf1)
newdf1.loc[1,'D'] = "Sneha"
print(newdf1)
print(newdf)
newdf.drop([1,5],axis=0,inplace=True)
print(newdf)
newdf.drop(['A','C'],axis=1,inplace=True)
print(newdf)
# Resetting the index:
print(newdf.reset_index())
newdf.reset_index(inplace=True,drop=True)
print(newdf)
