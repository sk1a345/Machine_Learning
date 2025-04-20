import pandas as pd
import numpy as np

df = pd.read_csv("Practice/Logistic_Regression.csv")
# print(df)

# print(df.info())
print(df.isnull().sum())

# Filling the null vlaues:

for col in df.columns:
    df[col].fillna(df[col].mode()[0],inplace=True)

# print(df.isnull().sum())
# print(df)

# diving data into input and output:
x = df.drop(columns=['EstimatedSalary','Purchased'],axis=1)
y = df['Purchased']

# print(x)
# print(y)

# print(y.value_counts())

#➡️➡️➡️ Performing scaling on the input:
from sklearn.preprocessing import StandardScaler

scl = StandardScaler()
x = scl.fit_transform(x) # <-- Converts it into a NumPy array
print("type(x): ",type(x))
x = pd.DataFrame(x,columns=['Age'])
print("type(x): ",type(x))


# ➡️➡️➡️ Splitting the data into training and teststing
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# ➡️➡️➡️ visualizing the data:
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x = x_train['Age'],y = y_train)
plt.title("Age vs Purchased(Before resampling)")
plt.xlabel("Age")
plt.ylabel("Purchased")
# plt.show()

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# ➡️➡️➡️ Applying the sampling (smote) on the data  

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=1)
x_resampled, y_resampled = smote.fit_resample(x_train,y_train)

# print(x_resampled.shape)
# print(y_resampled.shape)

# ➡️➡️➡️ Visualizing the data:

sns.scatterplot(x =x_resampled['Age'],y =y_resampled)
plt.title("Age vs Purchased(After resampling)")
plt.xlabel("Age")
plt.ylabel("Purchased")
# plt.show()

# ➡️➡️➡️➡️➡️You will observe that the the simply the 0 values has increased

# visualizing the outliers:
# # using the boxplot:
# plt.figure(figsize=(15,10))
# sns.boxplot(x=x_train['Age'],y=y_train)
# # plt.show()

#➡️➡️➡️  training the model:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


lg = LogisticRegression()
lg.fit(x_resampled,y_resampled)

y_pred = lg.predict(x_test)

# ➡️➡️➡️ Confusion matrics:
conf_mat = confusion_matrix(y_test,y_pred)
print("Confusion matrics is:\n",conf_mat)

# ➡️➡️➡️ Classification report:
clss_re = classification_report(y_test,y_pred)
print("Classification report:\n",clss_re)

# ➡️➡️➡️ Printing score:
sc = lg.score(x_test,y_test)
print("Score is: ",sc*100," %")

# ➡️➡️➡️Visualizing the data 
sns.scatterplot(x='Age',y='Purchased',data = df,color = 'blue')
sns.lineplot(x = 'Age',y = lg.predict(x),data = df,color = 'Red')
plt.show()

#➡️➡️➡️  Taking user input:
user_input = float(input("Enter your age: "))

user_input = scl.transform([[user_input]])
predicted_purchased = lg.predict(user_input)

print("Predicted purchaed is: ",predicted_purchased)