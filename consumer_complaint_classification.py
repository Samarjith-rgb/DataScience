import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load and perform EDA
def load_and_explore_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    return df

# 2. Text Pre-processing
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    return ''

# 3. Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Linear SVM': LinearSVC()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def main():
    # Load data
    df = load_and_explore_data('complaints.csv')
    
    # Assuming the columns are 'complaint_text' and 'category'
    # Adjust these column names based on your actual dataset
    X = df['complaint_text']
    y = df['category']
    
    # Preprocess text
    print("\nPreprocessing text...")
    X_processed = X.apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Vectorize text
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train_vec, X_test_vec, y_train, y_test)
    
    # Save the best model and vectorizer
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Example prediction function
    def predict_complaint(text):
        processed_text = preprocess_text(text)
        text_vec = vectorizer.transform([processed_text])
        prediction = best_model.predict(text_vec)
        return prediction[0]
    
    # Example prediction
    example_complaint = "I have been receiving calls from a debt collector about a credit card I don't recognize"
    prediction = predict_complaint(example_complaint)
    print(f"\nExample prediction for complaint: {example_complaint}")
    print(f"Predicted category: {prediction}")

if __name__ == "__main__":
    main() 
