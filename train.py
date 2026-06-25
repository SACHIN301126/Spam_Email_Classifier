import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert labels into numbers
# ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and labels
X = data['message']
y = data['label']

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Test accuracy
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Model Accuracy : {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")