import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
We are going to compare the performance of the model with different
number of rows of the dataset. We will use 1000, 2000 and 4000 rows
'''

def load_and_preprocess_data(filepath, nrows):
    df = pd.read_csv(filepath, nrows=nrows)
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].str.replace(r'[^\w\s]', '', regex=True)
    df['review'] = df['review'].str.replace(r'\d+', '', regex=True)
    
    # Tokenization and removal of stopwords
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    
    # Feature Engineering
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X = tfidf_vectorizer.fit_transform(df['review']).toarray()
    y = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    return X, y

# Load and preprocess data
X_1000, y_1000 = load_and_preprocess_data('data/processed/IMDB_Dataset_reduced.csv', 1000)
X_2000, y_2000 = load_and_preprocess_data('data/processed/IMDB_Dataset_reduced.csv', 2000)
X_4000, y_4000 = load_and_preprocess_data('data/processed/IMDB_Dataset_reduced.csv', 4000)

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

# Train and evaluate for 1000 rows
accuracy_1000, report_1000 = train_and_evaluate(X_1000, y_1000)

# Train and evaluate for 2000 rows
accuracy_2000, report_2000 = train_and_evaluate(X_2000, y_2000)

# Train and evaluate for 4000 rows
accuracy_4000, report_4000 = train_and_evaluate(X_4000, y_4000)

print("Accuracy for 1000 rows: ", accuracy_1000)
print("Report for 1000 rows:\n", report_1000)

print("Accuracy for 2000 rows: ", accuracy_2000)
print("Report for 2000 rows:\n", report_2000)

print("Accuracy for 4000 rows: ", accuracy_4000)
print("Report for 4000 rows:\n", report_4000)