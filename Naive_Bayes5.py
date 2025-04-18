# A company is looking to develop text classification model to categorize customer reviews into positive or negative sentiments.They want to use Naive Bayes classification to automatically analyze and classify the reviews they receive aiming to understand customer satisfaction levels and sentiments. The comapny describes a model that can accurately predict whther a customer review expresses a positive or negative sentiment.

# Step1:
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #converts text into numerical metrics
from sklearn.naive_bayes import MultinomialNB


# Step2
reviews =  ["The product is excellent and works perfectly",
            "The product is not good, very desappointing",
            "Terrible product and waste of money",
            "I love this product and it is amazing"
]

sentiments = np.array([1,0,0,1])

# Step 3:
vectorizer = CountVectorizer()
x =vectorizer.fit_transform(reviews)

#Step no 4:
classifier = MultinomialNB()
classifier.fit(x,sentiments)


# Step 5:
def classify_new_review(review):
    review_vectorized = vectorizer.transform([review])
    prediction = classifier.predict(review_vectorized)
    if prediction[0]==1:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"

# step 6:
user_input = input("Enter your review: ")
print(classify_new_review(user_input))



