import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the data
try:
    df = pd.read_csv("patient_health_dataset.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Print column names to verify
print("Columns in dataset:", df.columns)

# Create `input_text` by combining symptom-related columns
try:
    df['input_text'] = df[['Symptom_1', 'Symptom_2', 'Blood_Pressure']].astype(str).agg(' '.join, axis=1)
    df['response_text'] = df['Health_Condition']
    print("Created `input_text` and `response_text` columns.")
except KeyError as e:
    print(f"Error creating input_text and response_text: {e}")
    exit()

# Drop rows with missing values in the required columns
df.dropna(subset=['input_text', 'response_text'], inplace=True)
print(f"Dataset cleaned. Shape after dropping missing rows: {df.shape}")

# Preprocess text function
def process_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply preprocessing to the input column
df['Cleaned_input'] = df['input_text'].apply(process_text)

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set: {train_df.shape}, Testing set: {test_df.shape}")

# Vectorize the cleaned input text
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),  # Unigrams and bigrams
    dtype='float32'
)

# Create train and test matrices
try:
    train_matrix = vectorizer.fit_transform(train_df['Cleaned_input'])
    test_matrix = vectorizer.transform(test_df['Cleaned_input'])
    print(f"Train matrix shape: {train_matrix.shape}, Test matrix shape: {test_matrix.shape}")
except Exception as e:
    print(f"Error creating matrices: {e}")
    exit()

# Train Nearest Neighbors model
knn = NearestNeighbors(
    n_neighbors=1,
    algorithm='auto',
    metric='cosine'
)

try:
    knn.fit(train_matrix)
    print("Nearest Neighbors model trained successfully.")
except Exception as e:
    print(f"Error training model: {e}")
    exit()

# Save the vectorizer and model
try:
    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open('knn_model.pkl', 'wb') as model_file:
        pickle.dump(knn, model_file)
    print("Vectorizer and KNN model saved successfully.")
except Exception as e:
    print(f"Error saving models: {e}")

# Function to get response
def get_response(user_input, vectorizer, model, dataframe, top_k=1):
    processed_input = process_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    distances, indices = model.kneighbors(input_vector, n_neighbors=top_k)
    responses = dataframe['response_text'].iloc[indices[0]].tolist()
    return responses

# Calculate accuracy on the test dataset
def calculate_accuracy(test_df, vectorizer, model, train_df):
    correct_predictions = 0
    for _, row in test_df.iterrows():
        user_input = row['Cleaned_input']
        actual_response = row['response_text']
        predicted_response = get_response(user_input, vectorizer, model, train_df, top_k=1)[0]
        if predicted_response.strip().lower() == actual_response.strip().lower():
            correct_predictions += 1
    accuracy = correct_predictions / len(test_df)
    return accuracy

# Main program
if __name__ == "__main__":
    # Calculate accuracy
    print("Calculating accuracy on the test dataset...")
    accuracy = calculate_accuracy(test_df, vectorizer, knn, train_df)
    print(f"Model Accuracy: {accuracy:.2%}")

    # Chatbot interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        try:
            response = get_response(user_input, vectorizer, knn, train_df, top_k=1)
            print(f"Bot: {response[0]}")
        except Exception as e:
            print(f"Error generating response: {e}")
