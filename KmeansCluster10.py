# Problem Statement:A mall wants to understand its customer base better to improve its mareting strategy. The goal is to cluster customers into groups based on their annual income and spending score, so the marketing team can tailr their campaigns to each cluster's preferences and behaviors:


# step 1:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

# Generate Dataset:
np.random.seed(42)
data = {"AnnualIncome":np.random.randint(30000,100000,100),'SpendingScore':np.random.randint(1,100,100)}

df = pd.DataFrame(data)
# print(df.values)
# print(df)


plt.title("Customer Data -Annual Income vs Spending Score")
plt.scatter(df['AnnualIncome'],df['SpendingScore'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
# plt.show()


x = df.values
kmeans = KMeans(n_clusters=3,random_state=42)
df['Cluster'] = kmeans.fit_predict(x)
# print(df)

plt.scatter(df['AnnualIncome'],df['SpendingScore'],c = df['Cluster'],cmap = 'rainbow')
plt.title("KMeans clustering -anual income and spending score")
plt.ylabel("Spending Score")
plt.xlabel("Annual Income")
# plt.show()

# User Input:
income = int(input("Enter your annual income: "))
spend = int(input("Enter your spending score: "))

user_input = pd.DataFrame({'AnnualIncome':[income],'SpendingScore':[spend]})
user_cluster =kmeans.predict(user_input)
print(f"The user Belongs to Cluster: {user_cluster[0]}")