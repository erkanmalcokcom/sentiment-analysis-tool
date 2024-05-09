# Sentiment Analysis Model for IMDB Movie Reviews

Sentiment analysis is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. In this project, we build a sentiment analysis model to predict the sentiment of IMDB movie reviews as positive or negative.

The dataset consists of 50,000 IMDB movie reviews, with 25,000 labeled for training and 25,000 for testing. The reviews are preprocessed and converted into numerical features using TF-IDF vectorization. We then build a logistic regression model to classify the reviews into positive or negative sentiments.

The model achieves an accuracy of 86.93% on the test set, with balanced precision and recall for both positive and negative sentiments. This indicates that the model performs well across both classes and is equally good at identifying positive and negative sentiments.

model.py is the main script that loads and preprocesses the data, builds the sentiment analysis model, evaluates the model, and analyzes the results. The project also includes a requirements.txt file listing the required libraries to run the script.

reduce_csv_size.py is a script that reduces the size of the original dataset by randomly sampling a subset of reviews. This is useful for testing the model on a smaller dataset before running it on the full dataset.

To run the project, you can follow these steps:

1. Clone the repository:
```bash

git clone

```

2. Install the required libraries:
```bash

pip install -r requirements.txt

```

3. Run the model script:
```bash

python model.py

```

This will load the IMDB movie reviews dataset, preprocess the data, build the sentiment analysis model, evaluate the model on the test set, and analyze the results.

## Project Structure
The project is structured as follows:

1. **Data Preparation**: Load and preprocess the IMDB movie reviews dataset.
2. **Feature Engineering**: Convert text data into numerical features using TF-IDF vectorization.
3. **Model Building**: Build a logistic regression model to classify reviews into positive or negative sentiments.
4. **Model Evaluation**: Evaluate the model on the test set using accuracy, precision, recall, F1-score, and confusion matrix.
5. **Result Analysis**: Analyze the results of the sentiment analysis model and discuss next steps.

# Result Analysis of Sentiment Analysis Model
An accuracy of 0.8693 is quite suitable for sentiment analysis tasks, especially with a balanced dataset, as reflected by your numbers. At 86.93%, the model performs well across both classes.


- Precision and recall are well-balanced and very close to each other for both' negative' and' positive' sentiments, which is excellent. This suggests that your model is equally good at identifying positive and negative feelings and does not bias toward one class.

- Precision for both classes is around 87%, meaning that when the model predicts a particular sentiment, it is correct about 87% of the time.

- Recall is also balanced, with 86% for negative and 88% for positive. This indicates how well the model identifies all relevant instances of each sentiment. 

- The F1-score, the harmonic mean of precision and recall, is also balanced at around 87% for both classes. This score is critical as it better measures the incorrectly classified cases than the accuracy metric, especially if the classes are imbalanced.

- The confusion matrix shows that:
  - **True Negatives (TN)**: 4275 (correctly predicted negative)
  - **False Positives (FP)**: 686 (incorrectly predicted as positive)
  - **False Negatives (FN)**: 621 (incorrectly predicted as negative)
  - **True Positives (TP)**: 4418 (correctly predicted positive)

These numbers show that your model has a slightly higher number of false positives than false negatives, which might indicate a slight bias towards predicting positive sentiments, but this is minimal.

### Next Steps
1. **Further Tuning**: Depending on your specific requirements (e.g., reducing false positives or false negatives), you could further tune the model or threshold for classification.
2. **Error Analysis**: Dive deeper into the errors (both false positives and false negatives) to understand if there are specific types of reviews or phrases that are misclassified.
3. **Feature Enhancement**: Consider adding more nuanced features that might capture the sentiment more accurately, such as bigrams or trigrams in your TF-IDF representation or experimenting with different embeddings.
4. **Model Experimentation**: Try different models or ensemble methods to see if performance can be further improved.
5. **Real-world Application**: Consider deploying the model in a real-world scenario to get feedback on its performance in practical applications, which could guide further refinements.