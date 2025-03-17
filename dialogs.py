import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("patient_health_dataset.csv", engine='python', names=['input_text', 'response_text'], on_bad_lines='skip')
df.dropna(subset=['input_text', 'response_text'], inplace=True)

def process_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['Processed_input'] = df['input_text'].apply(process_text)
df['Processed_response'] = df['response_text'].apply(process_text)

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', dtype='float32')
x = vectorizer.fit_transform(df['Processed_input'])
y = df['Processed_response']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


 