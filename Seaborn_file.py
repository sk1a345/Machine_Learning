import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

'''
# data_set1 = sns.load_dataset("penguins")

var = [1,2,3,4,5,6]
var_1 = [2,3,4,5,6,7]
# plt.plot(var,var_1)
# plt.show() 

# x1 = pd.DataFrame({"var":var,"var_1":var_1})

# sns.lineplot(x = "var",y="var_1",data=x1)

data_set1 = sns.load_dataset("penguins").head(20)

# print(data_set1)
# style = "sex",palette="flag",
# legend parameters list: auto,brief,full,boolean

sns.lineplot(x ="bill_length_mm", y ="flipper_length_mm",data = data_set1,hue ="sex",style = "sex",size=10,markers = ["o",">"],dashes=False,legend="brief")
plt.grid()
plt.title("Python")
plt.show()
'''
'''
#Barplot:
# ci(line above the bar ci = value(inbetween 1-100))
dataset = sns.load_dataset("penguins")
# print(dataset.to_string())
order_1 = ["Dream","Torgersen","Biscoe"]
sns.barplot(x="island",y = "bill_length_mm",data = dataset,hue = "sex",order=order_1,hue_order = ["Female",'Male'],ci =30,orient="v",palette="icefire")
plt.show() 

'''

'''
# Histogram Plot:
data_s = sns.load_dataset("penguins")
# print(data_s)

sns.displot(data_s["flipper_length_mm"],bins=[170,180,190,200,210,220,230,240],kde=True,rug=True,color='k',log_scale=True)  
plt.show()'''
'''
# Scatter Plot:

scatter_var = sns.load_dataset("penguins").head(20)
# print(scatter_var)
sns.scatterplot(x="bill_length_mm",y="bill_depth_mm",data = scatter_var,markers=">",hue="sex",style="sex",sizes=(100,0),palette="icefire")
plt.show()'''

'''
#HeatMap:

# var = np.linspace(1,10,20).reshape(4,5)
# print(var)
# sns.heatmap(var)
# plt.show()

d = sns.load_dataset("anagrams")
x =d.drop(columns=["attnr"],axis=1).head(10)
print(d)
print(x)
sns.heatmap(x,vmin = 0,vmax =12,cmap = "Reds",annot = True)
plt.show()
'''
'''# Count plot:counts the no. of records by category. 
# Barplot plots a value or metric for each category(by default barplot plots the mean of a variable, by category)
data_c = sns.load_dataset("tips")
print(data_c.to_string())
sns.countplot(x="sex",data = data_c,hue="smoker",palette="bwr",color="r")
plt.show()
# sns.barplot(x="sex",y="size",data = data_c)
# plt.show()
'''
'''# Violine Plot:

vdata = sns.load_dataset("tips")
# print(vdata)
# sns.violinplot(x = "time",y = "total_bill",data = vdata,linewidth=3,palette='flag',linecolor="black",order = ['Dinner','Lunch'],saturation=1)

# sns.violinplot(x='day',y='total_bill',data = vdata,hue = "sex",split = True)
# sns.violinplot(x = "total_bill",y="day",data = vdata,hue = "sex",split = True)
# sns.violinplot(y=vdata["total_bill"],color="r")
plt.show()'''

'''
# Pair plot in Seaborn:

pdata = sns.load_dataset("tips")
# print(pdata)
# sns.pairplot(pdata,vars=["total_bill","tip"], hue= "sex",hue_order=["Male","Female"])
# sns.pairplot(pdata,hue="sex",hue_order=["Female","Male"],palette="BuGn",kind="kde",diag_kind="hist")
sns.pairplot(pdata,hue="sex",hue_order=["Female","Male"],markers=["*",">"],diag_kind="hist")
plt.show()
'''
'''
# Strip Plot

sdata = sns.load_dataset("tips")
# sns.boxplot(sdata)
sns.stripplot(x="day",y="total_bill",data = sdata,hue="sex")
plt.show()'''
'''
# Box_plot in Seaborn:
bdata = sns.load_dataset("tips")
sns.set(style="whitegrid")
# sns.boxplot(x="day",y="total_bill",data=bdata,hue="sex",color="g")
# sns.boxplot(y="day",x="total_bill",hue="sex",data = bdata,order=["Fri","Sun","Thur","Sat"],showmeans=True,meanprops={"marker":">",'markeredgecolor':"b"})  
sns.boxplot(data=bdata,orient="h")
plt.show()'''

'''# Factor Plot = catplot:
fplot = sns.load_dataset("tips")
# sns.factorplot(x="size",y="tips",data = fplot)
# Here factorplot function has been renamed to catplot:
sns.catplot(x="day", data = fplot,kind="count",palette="flag")
plt.show()'''
'''
# Styling plot in seaborn:
splot = sns.load_dataset('tips')
# print(splot)
# sns.set_style("dark")
sns.set_style("whitegrid")
sns.barplot(x="day",y="total_bill",data = splot,palette="Accent")
plt.grid()

sns.despine()# removing axis line

# plt.figure(figsize=(3,5))# increasing the figure size:
# sns.set_context("talk",font_scale=7)

plt.show()'''
'''
# Multiple Plots(Facet-Grid) In Seaborn
sdata = sns.load_dataset("tips")
# print(sdata)
# fg = sns.FacetGrid(sdata,col="sex")
# fg = sns.FacetGrid(sdata,col="sex",hue="day")
fg = sns.FacetGrid(sdata,col="day",hue="sex",palette="bwr")
# fg.map(plt.scatter,"total_bill","tip").add_legend()
fg.map(plt.bar,"total_bill","tip",edgecolor="r").add_legend()
plt.show() ''' 