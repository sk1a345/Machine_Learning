import numpy as np
import pandas as pd
# df =  { 'Names' :["Sneha","Kari","Yash","Abhi"]}
# s = ["Sneha","Sanju",'Kari','Yash']
# # print(pd.DataFrame(df))
# # newdf = pd.DataFrame(df)
# # print(newdf)
# # print(pd.__version__)
# newS = pd.Series(s,index = ['a','b','c','d'])
# print(type(newS))
# print(newS)
# print(type(newS)) 
# print(newS['a'])

# newD = pd.DataFrame(df)
# print(newD)
# print(type(newD))
# keyvalue:

# se = {"Sneha":22,"kari":15,'Abhi':22}
# newse = pd.Series(se)
# print(newse)
# print(newse['Sneha'])
# data = {
#     'Name':["Sneha","Kari","Yash","Abhi"],
#     'Marks':[100,200,300,400]
# }
# newd = pd.DataFrame(data)sssss
# print(newd)
# print(type(newd))
# s = [1,2,3,4]
# ns = pd.Series(s)
# print(ns)
# print(type(ns))
# newdf = pd.DataFrame(data)
# print(newdf)
# # print(newdf.loc[0])

# print(newdf.loc[[0,2]]) 
# print(newdf)
# n = pd.DataFrame(data,index=['a','b','c','d'])
# print(n)
# print(n.loc['b'])
# d = {"Country":["Japan","America","China","India"],
#      'pop':[100,200,300,400]
#     }
# newd = pd.DataFrame(d)
# print(newd)
# news = pd.Series(d)
# print(news)
# Tip: use to_string() to print the entire DataFrame.
# readcsv = pd.read_csv("Data.csv")
# # print(readcsv.to_string())
# # print(readcsv)
# print(pd.options.display.max_rows)
# print(pd.options.display.max_columns)
# # pd.options.display.max_rows = 909090
# print(readcsv)
# readjson =pd.read_json("dataj.json")
# print(readjson)
# pd.options.display.max_rows = 4567
# print(readjson)

'''# head() method:
h = pd.read_csv("Data.csv")
print(h.to_string())
# print(h.head(6))
# print(h.head(20))
# print(h.head())
# # print(h.head())
# print(h.tail())
# print(h.tail(5))
print(h.info())'''

# newfile = pd.read_csv("Data.csv")
# print(newfile.to_string())
# print(newfile.info())
# print(newfile.to_String())
# # 18 22 28:
# file = newfile.dropna()
# print(file.to_string())
# newfile.dropna(inplace = True)
# # print(newfile.info())
# newfile.fillna(1000,inplace=True)
# print(newfile.info())
# newfile['Calories'].fillna(1000,inplace = True)
# print(newfile.to_string())
# print(newfile.info())
# print(newfile.to_string())
# # mean()
# mean = newfile["Calories"].mean()
# print(mean)
# newfile['Calories'].fillna(mean,inplace=True)
# print(newfile.to_string())

# medi = newfile['Calories'].median()
# print(medi)
# newfile['Calories'].fillna(medi,inplace = True)
# print(newfile.to_string())

# mod = newfile['Calories'].mode()[0]
# print(mod)
# newfile['Calories'].fillna(mod,inplace=True)
# print(newfile.to_string())
# newfile = pd.read_csv('CleanData.csv')
# newfile['Date'] = pd.to_datetime(newfile['Date'])
# print(newfile.to_string())
df = pd.read_csv("CleanData.csv")
# print(df.to_string())
# df.loc[7,'Duration'] = 45
# print(df.to_string())
# print(df.index)
# for x in df.index:
#     if(df.loc[x,'Duration']<60):
#         # df.loc[x,'Duration'] = 1
#         df.drop(x,inplace=True)
# print(df.to_string())
# print(df.duplicated())
# print(df.to_string())
# df.drop_duplicates(inplace = True)
# print(df.to_string())
import matplotlib.pyplot as plt

# df = pd.read_csv("CleanData.csv")
# df.plot()
# plt.show()

# df = pd.read_csv("CleanData.csv")
# df.plot(kind ="scatter",x = "Duration", y ="Maxpulse")
# plt.show()

# df = pd.read_csv("CleanData.csv")
# df.plot(kind ="scatter" ,x = "Duration",y="Calories")
# plt.show()
df["Duration"].plot(kind = 'hist')
plt.show()