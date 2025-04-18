import numpy as np
import pandas as pd
import sklearn

url = "https://raw.githubusercontent.com/611noorsaeed/100-days-Scikit-Learn-Tutorials-/refs/heads/main/10%20pipe%20dataset.csv"

# df = pd.read_csv(url)
df = pd.read_csv('Pipeline.csv')
# print(df.head()) 

new_df = df.copy() #copying the original

new_df.drop("Time",inplace = True,axis=1) #dropping the Time column

# print(df)
# print(new_df)

# print(new_df['Accident_severity'].value_counts())

from sklearn.preprocessing import LabelEncoder

# creating the object of the LabelEncoder:
lb = LabelEncoder()
new_df['Accident_severity'] = lb.fit_transform(new_df['Accident_severity'])

# print(new_df['Accident_severity'].value_counts())

from imblearn.over_sampling import RandomOverSampler


x = new_df.drop('Accident_severity',axis=1) #all inputs except Accident_severity(output feature)
y = new_df['Accident_severity'] #output

oversampler = RandomOverSampler(random_state=1)
x_resampled, y_resampled = oversampler.fit_resample(x,y)

# print(y_resampled.value_counts()) 

# Train/Test/Split:
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_resampled,y_resampled, test_size= 0.2,random_state =42)
# print(new_df.isnull.sum())



from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Define the strategies for each column
strategies = {
    3: 'most_frequent',   # Educational_level
    4: 'most_frequent',   # Vehicle_driver_relation
    5: 'most_frequent',   # Driving_experience
    6: 'most_frequent',   # Type_of_vehicle
    8: 'constant',        # Service_year_of_vehicle
    9: 'constant',        # Defect_of_vehicle
    10: 'most_frequent',  # Area_accident_occured
    11: 'most_frequent',  # Lanes_or_Medians
    12: 'most_frequent',  # Road_allignment
    13: 'most_frequent',  # Types_of_Junction
    14: 'most_frequent',  # Road_surface_type
    18: 'most_frequent',  # Type_of_collision
    21: 'most_frequent',  # Vehicle_movement
    26: 'most_frequent',  # Work_of_casuality
    27: 'most_frequent'   # Fitness_of_casuality
}

# Create a ColumnTransformer for data preprocessing
tf1 = ColumnTransformer([
    ('impute_educational_level', SimpleImputer(strategy=strategies[3]), [3]),
    ('impute_Vehicle_driver_relation', SimpleImputer(strategy=strategies[4]), [4]),
    ('impute_Driving_experience', SimpleImputer(strategy=strategies[5]), [5]),
    ('impute_Type_of_vehicle', SimpleImputer(strategy=strategies[6]), [6]),
    ('impute_Service_year_of_vehicle', SimpleImputer(strategy=strategies[8], fill_value='Unknown'), [8]),
    ('impute_Defect_of_vehicle', SimpleImputer(strategy=strategies[9], fill_value='Unknown'), [9]),
    ('impute_Area_accident_occured', SimpleImputer(strategy=strategies[10]), [10]),
    ('impute_Lanes_or_Medians', SimpleImputer(strategy=strategies[11]), [11]),
    ('impute_Road_allignment', SimpleImputer(strategy=strategies[12]), [12]),
    ('impute_Types_of_Junction', SimpleImputer(strategy=strategies[13]), [13]),
    ('impu3hte_Road_surface_type', SimpleImputer(strategy=strategies[14]), [14]),
    ('impute_Type_of_collision', SimpleImputer(strategy=strategies[18]), [18]),
    ('impute_Vehicle_movement', SimpleImputer(strategy=strategies[21]), [21]),
    ('impute_Work_of_casuality', SimpleImputer(strategy=strategies[26]), [26]),
    ('impute_Fitness_of_casuality', SimpleImputer(strategy=strategies[27]), [27])
], remainder='passthrough')

# Encode Categorical Columns

from sklearn.preprocessing import OneHotEncoder
#define the objet columns indices
object_columns_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
 
tf2 = ColumnTransformer([
    (f'ohe_{col}',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[col])
    for col in object_columns_indices
],remainder = 'passthrough')


# Feature selection using chi2 statistic

from sklearn.feature_selection import SelectKBest,chi2
tf4 = SelectKBest(chi2,k=10)


# Model(Random forest Classifier)

from sklearn.ensemble import RandomForestClassifier

tf5 = RandomForestClassifier()

# Create Piplene

from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('trf1',tf1),
    ('trf2',tf2),
    ('trf4',tf4),
    ('trf5',tf5)
])
#Train the pipeline:
p =pipe.fit(x_train,y_train)
# print(p)

# Accuracy Score:
# predict:
from sklearn.metrics import accuracy_score

y_pred = pipe.predict(x_test)
s =accuracy_score(y_test,y_pred)
print(s)



