import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf

# Ensure you have the necessary NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')


# Example DataFrame
data = {'review': ['This movie was great!', 'I did not like the movie', 'Just okay, nothing special', 'Fantastic plot and acting', 'Worst movie ever'],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']}


# add new data
new_data = pd.read_csv('data/processed/IMDB Dataset_reduced.csv')
print(new_data.head())
data.update(new_data)


df = pd.DataFrame(data)


# Preprocessing

df['review'] = df['review'].str.lower()
df['review'] = df['review'].str.replace(r'[^\w\s]', '')
df['review'] = df['review'].str.replace(r'\d+', '')

# Tokenization
stop_words = set(stopwords.words('english'))
df['review']

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(df['review']).toarray()
y = df['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')