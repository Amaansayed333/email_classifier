import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

def clean_text(text):
    # Remove leading/trailing whitespace and normalize multiple spaces
    return re.sub(r'\s+', ' ', str(text).strip())

def train_and_save_model(csv_path="combined_emails_with_natural_pii.csv"):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Clean the email text
    df['email_clean'] = df['email'].apply(clean_text)

    # Encode the target labels (email type) as integers
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['type'])

    # Convert text data to TF-IDF feature vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['email_clean'])
    y = df['label_encoded']

    # Split dataset into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save the model, vectorizer, and label encoder to disk
    joblib.dump(clf, "naive_bayes_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("Models saved.")
    return clf, vectorizer, label_encoder

def load_models():
    # Load previously saved model components
    clf = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return clf, vectorizer, label_encoder

def predict_email_category(email, vectorizer, clf, label_encoder):
    # Predict email category given a raw email string
    vec = vectorizer.transform([email])
    pred = clf.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

if __name__ == "__main__":
    train_and_save_model()
