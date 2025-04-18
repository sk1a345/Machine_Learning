import matplotlib.pyplot as plt #matplotlib.pyplot is a module in the Matplotlib library
import numpy as np

# matplotlib.pyplot: This is a module in the Matplotlib library used for plotting graphs and charts.

# as plt: This assigns an alias to matplotlib.pyplot, allowing us to use plt instead of typing matplotlib.pyplot every time.
'''
# printing the versions
print(plt.__version__) 
print(plt._VersionInfo)
# print(np.__verison__)
print(np.version.version)
print(np.__array_api_version__)
print(np.__version__)'''
'''
# Graph1:
# Define data points
xpoints = np.array([1,2,3,4,5])
ypoints = np.array([1,2,3,4,5])

# Plot the points
plt.plot(xpoints, ypoints)
# plt.savefig("plot.png")  #saves the image as png

# Show the plot (Works in VS Code if using an interactive backend)
# plt.show()
# print(plt.show()) ->output None'''

'''# Graph2:
xpoint = np.array([5,6])
ypoint = np.array([5,6])
plt.plot(xpoint,ypoint,'o') #Plotting Without Line To plot only the markers, you can use shortcut string notation parameter 'o', which means 'rings'.
plt.show()
'''
'''
# Graph3:
xpoint = np.array([1,3,5,7])
ypoint = np.array([2,4,6,8])
plt.plot(xpoint,ypoint)
plt.show()'''

'''# Graph4:
xpoint = np.array([1, 2, 6, 8])
ypoint = np.array([3, 8, 1, 10])
plt.plot(xpoint,ypoint)
plt.show()
'''
'''#  Default X-Points
# Graph5
ypoint = np.array([3,8,1,10])  
plt.plot(ypoint) #here bydefault xpoints as [1,2,3,4....] will be taken
plt.show() '''

'''# Markers:
#Graph6:
ypoint = np.array([3,8,1,10])
# plt.plot(ypoint,'o') #without graph line
# plt.plot(ypoint,marker = 'o')  #ring or circle#with graph line and the marker point will be printed
# plt.plot(ypoint,marker = '*') #marker as *
# plt.plot(ypoint,marker = '|') #marker as Vline
# plt.plot(ypoint,marker = '1') #marker as tri down
# plt.plot(ypoint,marker = '2') #marker as tri up
# plt.plot(ypoint,marker = '3') #marker as tri left
# plt.plot(ypoint,marker = '4') #marker as tri right
# plt.plot(ypoint,marker = '_') #marker as Hline
# plt.plot(ypoint,marker = '.') #marker as point
# plt.plot(ypoint,marker = ',') #marker as pixel
# plt.plot(ypoint,marker = '+') #marker as plus
# plt.plot(ypoint,marker = 'X') #marker as Xfilled
# plt.plot(ypoint,marker = 'x') #marker as X
# plt.plot(ypoint,marker = 'P') #marker as +filled
# plt.plot(ypoint,marker = 's') #marker as square
# plt.plot(ypoint,marker = 'D') #marker as Dimond
# plt.plot(ypoint,marker = 'd') #marker as dimond-thin
# plt.plot(ypoint,marker = 'p') #marker as pentagon
# plt.plot(ypoint,marker = 'H') #marker as Hexagon (left-right)
# plt.plot(ypoint,marker = 'h') #marker as Hexagon(up-down)
# plt.plot(ypoint,marker = 'v') #marker as trianle down
# plt.plot(ypoint,marker = '^') #marker as trianle up
# plt.plot(ypoint,marker = '<') #marker as trianle left
# plt.plot(ypoint,marker = '>') #marker as trianle right

plt.show()'''
'''# Format Strings fmt
#➡️➡️➡️➡️➡️ marker|line|color
ypoints = np.array([3,8,1,10])
# plt.plot(ypoints,'o:') #bydefault blue color and dotted line(:)
# plt.plot(ypoints,'o:r') #red color and dotted line(:)
# plt.plot(ypoints,'o-b') #blue color and solid line(-)
# plt.plot(ypoints,'o--c') #cyan color and dashed line(--)
# plt.plot(ypoints,'o-.m') #magenta(similar to purple) color and dashed-dotted line(-.)

# Markersize (markersize/ms = value) 
# plt.plot(ypoints,'.-g',markersize = 30)
# plt.plot(ypoints,'.-g',ms = 30)
# markeredgecolor(markeredgecolor/mec= color)
# plt.plot(ypoints,'.-r',ms = 20,mec = 'k') #mec = black
# markerfacecolor(markerfacecolor/mfc = color)
# plt.plot(ypoints,'.-r',ms = 30,mec = 'b',mfc = 'y')
# plt.plot(ypoints,'.-g',ms = 30,mec = 'hotpink',mfc = 'hotpink')
# plt.plot(ypoints,'.-r',ms = 30,mec = '#4CAF50',mfc = '#4CAF50') #hexadecimal-color
plt.show()'''

# Line constrution

#lineStyle
ypoints = np.array([1,8,3,10])
# plt.plot(ypoints,linestyle = 'dashed') #linestyle ls (dashed --)
# plt.plot(ypoints,ls = 'solid') #(solid -)
# plt.plot(ypoints,ls = 'dotted') #(dotted (:))
# plt.plot(ypoints,ls = 'dashdot') #(dashdot (-.))
# plt.plot(ypoints,ls = 'None',marker = 'o')

# Line color:

# plt.plot(ypoints,color = 'red')
# plt.plot(ypoints,color = 'hotpink')
# plt.plot(ypoints,color = 'pink')
# plt.plot(ypoints,color = 'k') #black
# plt.plot(ypoints,color = 'g') #green
# plt.plot(ypoints,color = '#4CAF50') #light green (hexadecimal color form)

# Line-Width:
# plt.plot(ypoints,lw = '20.5')
# plt.plot(ypoints,linewidth = '100')

'''# plotting multiple lines:
l1 = np.array([1,6,3,5])
l2 = np.array([8,4,6,1])
plt.plot(l1,color = 'red')
plt.plot(l2,color = 'green')
plt.show()'''
'''
# plotting the graph by the follwing way
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])
plt.plot(x1,y1,x2,y2)
plt.show()'''

'''# Creating the lables for the graph:
# Graph1:
x = np.array(['Sneha','mansi',"kari",'shagun','kiran'])
y = np.array([20,10,60,40,50])
plt.plot(x,y)
plt.xlabel("Names")
plt.ylabel("Marks")
plt.show()

# Graph2
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x,y)
plt.xlabel('Sports')
plt.ylabel("Players")
plt.title("Analysis of Sports")
plt.show()

# Graph3(Adding the fonts to the lables)
x = np.array(['sneha','kari','yash','abhi','aaru'])
y = np.array([1,2,3,4,5])
font1 = {'family':'serif','color':"Green",'size':20}
font2 = {'family':'serif','color':'Red','size':15}
plt.xlabel("Names",fontdict = font2)
plt.ylabel("Heights",fontdict = font2)
plt.title("Analysis of Heights",fontdict=font1)
plt.plot(x,y,'o-k') #marker|line|color(black-k)
plt.show()'''
'''
➡️➡️➡️Position the Title
You can use the loc parameter in title() to position the title.

Legal values are: 'left', 'right', and 'center'. Default value is 'center'.'''
'''
# Graph4:
sub = np.array(['Math','Os','ED',"IDA"]) 
mark = np.array([100,90,70,80])
font1 = {'family':'serif','color':"Green",'size':20}
font2 = {'family':'serif','color':"Red",'size':15}
plt.title("Result",loc = "right",fontdict=font1) #loc = left,right,center
plt.xlabel("Subjects",fontdict=font2)
plt.ylabel("Marks",fontdict=font2)
plt.plot(sub,mark,'o-b')
plt.show()'''


'''
# Graph5:
state = np.array(["MH",'UP',"MP",'GOA','JK'])
pop = np.array([100,20,40,150,30])
f1 = {'family':'serif','size':22,'color':"green"}
f2 = {'family':'serif','size':15,'color':"Red"}
plt.xlabel("States",fontdict=f2)
plt.ylabel("Population",fontdict = f2)
plt.title("Population Distribution Graph",fontdict=f1)
plt.plot(state,pop,'o-',color='hotpink')
plt.grid()   
plt.show()

# Graph6:
state = np.array(["MH",'UP',"MP",'GOA','JK'])
pop = np.array([100,20,40,150,30])
f1 = {'family':'serif','size':22,'color':"green"}
f2 = {'family':'serif','size':15,'color':"Red"}
plt.xlabel("States",fontdict=f2)
plt.ylabel("Population",fontdict = f2)
plt.title("Population Distribution Graph",fontdict=f1)
plt.plot(state,pop,'o-',color='hotpink')
# plt.grid(axis='x')   #Display  grid lines for the x-axis only:  
# plt.grid(axis='y') ##Display  grid lines for the y-axis only:  

# Set the line properties of the grid:
plt.grid(color = 'green',linewidth='0.7',linestyle='--')  
plt.show() '''

# subplots:
#Graph1:
'''x = np.array([1,2,3,4,5])
y = np.array([4,3,5,1,0])

plt.subplot(1,2,1)
plt.plot(x,y)

x = np.array([0,1,4,2,5])
y = np.array([1,2,3,4,0])

plt.subplot(1,2,2)
plt.plot(x,y)

plt.show()'''
'''
# Graph2:
x = np.array([1,2,3,4,5])
y = np.array([4,3,5,1,0])

plt.subplot(2,1,1)
plt.plot(x,y)

x = np.array([0,1,4,2,5])
y = np.array([1,2,3,4,0])

plt.subplot(2,1,2)
plt.plot(x,y)

plt.show()'''
'''
# Graph3:
dict = {'family':'serif','color':"hotpink",'fontsize':20} 
plt.suptitle("configuration of Graphs",fontdict = dict) #supertitle
x = np.array([1,2,3,4,5])
y = np.array([4,3,5,1,0])
plt.subplot(2,4,1)
plt.plot(x,y)

x = np.array([0,1,4,2,5])
y = np.array([1,2,3,4,0])
plt.subplot(2,4,2)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2,4,3)
plt.plot(x,y)

x = np.array([0,1,4,2,5])
y = np.array([1,2,3,4,0])
plt.subplot(2,4,4)
plt.plot(x,y)
x = np.array([1,2,3,4,5])
y = np.array([4,3,5,1,0])
plt.subplot(2,4,5)
plt.plot(x,y)

x = np.array([0,1,4,2,5])
y = np.array([1,2,3,4,0])
plt.subplot(2,4,6)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2,4,7)
plt.plot(x,y)

x = np.array([0,1,4,2,5])
y = np.array([1,2,3,4,0])
plt.subplot(2,4,8)
plt.plot(x,y)
plt.show()

# Graph
# Adding the title:
x = np.array([1,2,3,5])
y = np.array([3,2,1,0])

plt.subplot(1,2,1)
plt.plot(x,y)
plt.title("First")

x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

plt.subplot(1,2,2)
plt.plot(x,y)
plt.title("Second")

plt.show()

'''

# Drawing the Graphs:
'''
x = np.array(["Sneha",'Mansi','Shagun','Teju'])
sal = np.array([40,10,23,33])
plt.bar(x,sal)
plt.show()

# If you want the bars to be displayed horizontally instead of vertically, use the barh() function:
x = np.array(['A','B','C','D'])
y = np.array([10,20,30,40])
plt.barh(x,y)
plt.show()

x = np.array(['Nagpur','Amravati','Pune','Nashik','mumbai'])
y = np.array([100,29,45,90,67])
# plt.bar(x,y,color='hotpink')
# plt.bar(x,y,color='#4CAF50',width=0.5)
plt.barh(x,y,color='hotpink',height=0.5)
plt.show()
'''

'''# histogram creation:
x = np.random.normal(170, 10, 250)

plt.hist(x, bins=30, edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Normal Distribution (mean=170, std=10)")
plt.show()
'''
'''# Piechart creation
# By default the plotting of the first wedge starts from the x-axis and moves counterclockwise:
x = np.array([1,2,3,4])
plt.pie(x)
plt.show() '''
'''# Labels
# Add labels to the pie chart with the labels parameter.
# The labels parameter must be an array with one label for each wedge:
marks = np.array([30,40,15,16])
lable = ["kari",'sneha','shagun','mansi']
# print(type(lable))
plt.pie(marks,labels=lable)
plt.show()
'''
'''# Start Angle
# As mentioned the default start angle is at the x-axis, but you can change the start angle by specifying a startangle parameter.
# The startangle parameter is defined with an angle in degrees, default angle is 0:
sal = np.array([11,22,33,44]) 
labl = ['Apple','Mango',"Strowberry",'Chiku']
plt.pie(sal,labels=labl,startangle=90)
plt.show()
'''
'''# Explode
# Maybe you want one of the wedges to stand out? The explode parameter allows you to do that.
# The explode parameter, if specified, and not None, must be an array with one value for each wedge.
# Each value represents how far from the center each wedge is displayed:

marks = np.array([100,200,300,400])
sub = ['Maths','operating System','Python',"Java"]
ex = [0,0,0.2,0]
plt.pie(marks,labels=sub,explode= ex)
plt.show()'''
''' 
# Shadow
# Add a shadow to the pie chart by setting the shadows parameter to True:
pop = np.array([1,2,3,4])
country = ['japan','uk','china','India']
ex = [0,0,0,0.2]
plt.pie(pop,labels=country,shadow=True,explode=ex)
plt.show()'''

# n = np.array(['sneha','kari','yash'])
# arr = [1,2,3]
# plt.pie(n,labels=arr)
# plt.show() #ValueError: could not convert string to float: np.str_('sneha')
'''
# Colors
# You can set the color of each wedge with the colors parameter.
# The colors parameter, if specified, must be an array with one value for each wedge:

fruit_p = np.array([20,30,11,4])
fruit_name = ['Mango','Apple','Peru','Berry']
fruit_color = ['Orange','Red','Green','Black']
plt.pie(fruit_p,labels=fruit_name,colors=fruit_color)
plt.title("Fruits prize",color='Blue',fontsize=20)

# Legend
# To add a list of explanation for each wedge, use the legend() function:
plt.legend()
plt.show()
'''
'''# Legend With Header
# To add a header to the legend, add the title parameter to the legend function.

runs = np.array([90,100,200,300,50])
player = ['Virat','Mahi','Surya','Rohit','Hardik']
color = ['Red','Yellow','Blue','silver','Black']
ex = [0.2,0,0,0,0]
plt.pie(runs,labels=player,colors=color,explode=ex,shadow=True)
plt.legend(title="Five Player")
plt.title("Runs Classification",color="Green",size = 20)
plt.show()
'''