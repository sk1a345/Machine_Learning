import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download the required tokenizer
# nltk.download('punkt')
df = pd.read_csv('smsSpam.csv')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove digits and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove URLs
    text = re.sub(r'https\S+', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords:
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    

    # Initialize Porter Stemmer:
    st = PorterStemmer()
    # Perform Stemming:
    stemmed_words = [st.stem(word) for word in filtered_words]

    # Join the stemmed words back into a single string:
    cleaned_text = ' '.join(stemmed_words)
    return cleaned_text


# Example text
text_example = "This Dear is my  I SMS am loving my India 12343553 #@$%^&* https://noor.com"
# print(clean_text(text_example))

df['clean_text'] = df['sms'].apply(lambda x:clean_text(x))
# print(df.head())

# print(df['label'].value_counts())

# Balancing Dataset:
from imblearn.over_sampling import RandomOverSampler

# Assuming df is your Dataframe containing the data:
x = df.drop('label',axis=1) #features
y = df['label'] #ouput

# Initialize the resampling technique:
oversampler = RandomOverSampler()

# Perform resampling:
x_resampled,y_resampled = oversampler.fit_resample(x,y)

# Create a new DataFrame for the balanced dataset:
df_balanced = pd.DataFrame(x_resampled,columns=x.columns)

# Train test split and vectorization (TFIDF):
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Train test split:
x_train,x_test,y_train,y_test = train_test_split(x_resampled['clean_text'],y_resampled,test_size=0.2,random_state=1)

# TF-IDF vectorization:
tfidf_vectorizer = TfidfVectorizer()
x_train_t = tfidf_vectorizer.fit_transform(x_train)
x_test_t = tfidf_vectorizer.transform(x_test)


# ModelBuilding and ensemble learning:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Random forest Classifier:
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(x_train_t,y_train)
# Prediction:
y_pred = rf_classifier.predict(x_test)

# confusion matrix:
conf_matrix = confusion_matrix(y_test,y_pred)
class_report = classification_report(y_test,y_pred)
print("Confusion matrix : ",conf_matrix)

print("Classification Report: ",class_report)


