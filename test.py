import pandas as pd;
'''# dict = {"Sneha":[100],"Kartik":[200],"Abhi":[300],"Yash":[400]}
# print(type(dict))
# print(dict)
# result = pd.DataFrame(dict)
# print(result)
s = [1,2,3]
var = pd.Series(s) #here 0,1,2 will be the lables:
print(var)
print(var[2])
print(type(var))

# Creating your own label:

mylabel = pd.Series(s,index = ['x','y','z']) #here x,y,z will be the lables
print(mylabel)
print(mylabel["x"])

# Dictionary:
dict1 = {"Sneha": 19,"kartik":17,"Abhi":23,"Yash":22}
print(pd.Series(dict1))
# df = pd.DataFrame(dict1)
# print(df)
# Kyes of the dictionaries becomes the lables:
print(dict1["Sneha"])
print(dict1["Abhi"])

# To select only some of the items in the dictionary, use the index argument and specify only the items you want to include in the Series

dict3 = {"Day1":333,"Day2":345,"Day3": 45}
print(pd.Series(dict3))
newdic3 = pd.Series(dict3,index=["Day1","Day2"])

print(newdic3)'''
'''
# DataFramesData sets in Pandas are usually multi-dimensional tables, called DataFrames.
# DataFrames Examples:
dict4 = {"Pune":[4,3,2,1],"Nashik":[90,70,60,50],"Nagpur":[11,22,33,44],"Delhi":[1,2,3,4]}
newDict4 = pd.DataFrame(dict4,index=['a.','b.','c.','d.'])
print(newDict4)
# loc() variables
print(pd.DataFrame(dict4))
print(newDict4.loc['a.'])
print(newDict4.loc[['a.','b.']])
print(pd.DataFrame(dict4).loc[0])

import pandas as pd

dict5 = {0: "apple", 1: "banana", 2: "cherry"}

print(pd.Series(dict5).loc[0])
print(pd.Series(dict5))'''

# Load Files Into a DataFrame => If your data sets are stored in a file, Pandas can load them into a DataFrame.
'''
file1 = pd.read_csv("Data.csv")
print(file1)
print(pd.options.display.max_rows)
print(pd.options.display.max_columns)
pd.options.display.max_rows = 2222
# file2 = pd.read_csv("Data.csv")
# print(file2) #displays whole data
print(file1)
print(pd.options.display.max_rows)'''

'''# reading the Json file 
jfile = pd.read_json("dataj.json")
print(jfile)

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}
dataF = pd.DataFrame(data)
print(dataF)'''

'''newcsv = pd.read_csv("newCSV.csv")
# print(newcsv)

# print(newcsv.head())
# print(newcsv.head(15))
print(newcsv.tail())
print(newcsv.tail(10))
print(newcsv.info())'''

df = pd.read_csv('CleanData.csv')
'''
# Removing the expty cells =>One way to deal with empty cells is to remove rows that contain empty cells
new_df = df.dropna() #used to remove the expty cells

print(new_df.to_string()) ##Notice in the result that some rows have been removed (row 18, 22 and 28).
print(df) 
df.dropna(inplace=True)
print(df)'''

'''#Replacing the empty values:Another way of dealing with empty cells is to insert a new value instead.his way you do not have to delete entire rows just because of some empty cells.

# fillna()
# fillReturn = df.fillna(9999)
# print(fillReturn)

# df.fillna(1111,inplace=True)
# print(df)'''

'''# Replace Only For Specified Columns
df['Calories'].fillna("Sneha",inplace=True)
print(df)'''

# REplacing the expty cells with the mean of that column:
#Mean:
'''
x = df['Calories'].mean()
print(x)
df['Calories'].fillna(x,inplace=True)
print(df)'''
'''
#Median:
y = df['Calories'].median()
print(y)
df["Calories"].fillna(y,inplace=True)
print(df)'''

'''# Mode:
z = df['Date'].mode()[0] #The [0] is used to extract the first mode from the result of df['Date'].mode().
print(z) 
df['Date'].fillna(z,inplace=True)
print(df)
 
w = df['Calories'].mode()
print(w)
'''
'''# Cleaning data of the wrong foramte:
# In our Data Frame, we have two cells with the wrong format. Check out row 22 and 26, the 'Date' column should be a string that represents a date:

df['Date'] = pd.to_datetime(df['Date'],errors='coerce') 
df['Date'].fillna(pd.Timestamp('29/01/2006'),inplace=True)
print(df.to_string())'''

# Replacing Values
# df.loc[7,'Duration'] = 45
# print(df)

# for x in df.index:
#     if(df.loc[x,'Duration']>120):
#         df.loc[x,'Duration'] = 120
# print(df)

# Removing the rows
'''for x in df.index:
    if(df.loc[x,'Duration']>120): 
        df.drop(x,inplace=True) # it will drop that particular row
print(df)'''

'''#Removing the duplicates:
# By taking a look at our test data set, we can assume that row 11 and 12 are duplicates.
print(df.duplicated())

df.drop_duplicates(inplace=True)
print(df)'''
# Correlation

# The number varies from -1 to 1.
# 1 means that there is a 1 to 1 relationship (a perfect correlation), and for this data set, each time a value went up in the first column, the other one went up as well.

# 0.9 is also a good relationship, and if you increase one value, the other will probably increase as well.

# -0.9 would be just as good relationship as 0.9, but if you increase one value, the other will probably go down.

# 0.2 means NOT a good relationship, meaning that if one value goes up does not mean that the other will.
df1 = pd.read_csv('corr.csv')
# print(df1.to_string()) #displays the whole data
print(df1.corr())
