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
import tensorflow as tf  # Not strictly necessary for this code

from sklearn.preprocessing import label_binarize

def download_nltk_packages(packages):
    for package in packages:
        try:
            nltk.data.find('tokenizers/' + package) if package == 'punkt' else nltk.data.find('corpora/' + package)
        except LookupError:
            nltk.download(package)

# Specify the packages you need
packages = ['punkt', 'stopwords']

# Download the packages
download_nltk_packages(packages)

def preprocess_data(df):
    # Preprocessing
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].str.replace(r'[^\w\s]', '')
    df['review'] = df['review'].str.replace(r'\d+', '')

    # Tokenization
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

    # Feature Engineering
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X = tfidf_vectorizer.fit_transform(df['review']).toarray()
    y = df['sentiment'].map({'negative': 0, 'positive': 1})

    return X, y


# Example DataFrame
df = pd.read_csv('data/processed/IMDB_Dataset_reduced.csv')
df = df.iloc[:1000]  # For demonstration purposes

X, y = preprocess_data(df)

# Binarize the target variable (optional for multi-class classification)
# y = label_binarize(y, classes=['negative', 'neutral', 'positive'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Logistic Regression): {accuracy}')

# Optional: Evaluation with Thresholding on Probabilities
probabilities = model.predict_proba(X_test)
threshold = 0.6  # Adjust this threshold based on your needs

predictions = (probabilities[:, 1] >= threshold).astype(int)
print(f"Accuracy with Threshold ({threshold}): {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))
print("*" * 70)

# GridSearchCV for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}  # Logistic Regression parameters

# Create a GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5, scoring='accuracy', error_score='raise')

grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("*" * 70)

# Ensemble Model (Optional)
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Create different models (consider other options as well)
model1 = LogisticRegression(solver='saga')
model2 = SVC(probability=True)
model3 = RandomForestClassifier()

# Create a voting classifier
voting_classifier = VotingClassifier(
    estimators=[('lr', model1), ('svc', model2), ('rf', model3)],
    voting='soft')  # Soft voting returns the class label as argmax of the sum of predicted probabilities

# Fit the model
voting_classifier.fit(X_train, y_train)

# Predictions
y_pred_voting = voting_classifier.predict(X_test)

# Accuracy

accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f'Accuracy (Voting Classifier): {accuracy_voting}')
print(classification_report(y_test, y_pred_voting))
print("*" * 70)

# Function to perform the full model setup and evaluation
def run_experiment(nrows):
    df = pd.read_csv('data/processed/IMDB_Dataset_reduced.csv', nrows=nrows)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Set up and train Voting Classifier
    voting_classifier.fit(X_train, y_train)
    y_pred_voting = voting_classifier.predict(X_test)
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    
    return accuracy, accuracy_voting

# Run experiments for 1000 and 2000 rows
accuracy_1000, accuracy_voting_1000 = run_experiment(1000)
accuracy_2000, accuracy_voting_2000 = run_experiment(2000)

print(f"Results for 1000 rows - Logistic Regression: {accuracy_1000}, Voting Classifier: {accuracy_voting_1000}")
print(f"Results for 2000 rows - Logistic Regression: {accuracy_2000}, Voting Classifier: {accuracy_voting_2000}")
print("*" * 70)

