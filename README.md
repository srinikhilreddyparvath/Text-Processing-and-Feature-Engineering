# Text-Processing-and-Feature-Engineering
This repository demonstrates text processing techniques like tokenization, stopword removal, stemming, and lemmatization. It also covers feature engineering methods such as TF-IDF and word embeddings for building ML models or information retrieval systems.


Dataset:
I started with a simple dataset containing text data and corresponding labels for a sentiment analysis task. This dataset includes sentences expressing either positive or negative sentiments.

import pandas as pd

# Sample dataset
data = {'text': ["I love this product!", "This is the worst service ever.", "Absolutely fantastic experience.", 
                 "I hate waiting in long lines.", "The food was delicious but the service was slow."],
        'label': [1, 0, 1, 0, 1]}

df = pd.DataFrame(data)


Text Preprocessing:
To clean the text data, I applied several text processing techniques, including tokenization, stopword removal, and stemming/lemmatization. These steps are essential for reducing noise and standardizing the text before feeding it into a machine learning model.

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Stemming
    words_stemmed = [stemmer.stem(word) for word in words]
    
    # Lemmatization
    words_lemmatized = [lemmatizer.lemmatize(word) for word in words_stemmed]
    
    return " ".join(words_lemmatized)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)


Feature Engineering:
Next, I focused on feature engineering by using TF-IDF, a common method for converting text into numerical vectors. This step transforms the cleaned text data into a format that can be easily interpreted by machine learning algorithms.

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform the cleaned text data into TF-IDF features
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Labels
y = df['label']


Building a Simple Model:
To demonstrate the effectiveness of the preprocessing and feature engineering, I built a simple logistic regression model. I split the data into training and testing sets, trained the model on the training data, and then evaluated its performance on the test set.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


Summary:
Through this project, I’ve implemented a complete workflow for text processing and feature engineering in NLP, followed by building a machine learning model to analyze sentiment. The process involved cleaning the text data using tokenization, stopword removal, stemming, and lemmatization, and then transforming it into TF-IDF features. Finally, I built and evaluated a logistic regression model, which demonstrated the effectiveness of the preprocessing steps.

This project allowed me to explore various NLP techniques and understand their impact on the performance of downstream machine learning tasks.








