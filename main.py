import os
import re
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

df_path = 'patient_health_dataset.csv'
vectorizer_path = 'tfidf_vectorizer.pkl'
model_path = 'random_forest_model.pkl'

if not os.path.exists(df_path):
    raise FileNotFoundError("Error: 'patient_health_dataset.csv' is missing. Ensure it is in the correct directory.")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Error: 'tfidf_vectorizer.pkl' is missing. Train your model and save the vectorizer.")
if not os.path.exists(model_path):
    raise FileNotFoundError("Error: 'random_forest_model.pkl' is missing. Train your model and save the KNN model.")

df = pd.read_csv(df_path, sep='\t', header=None, names=['input_text', 'response_text'], on_bad_lines='skip')
df.dropna(subset=['input_text', 'response_text'], inplace=True)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(model_path, 'rb') as f:
    random_forest_model = pickle.load(f)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def get_response(user_input, vectorizer, model, dataframe):
    processed_input = preprocess_text(user_input)
    try:
        input_vector = vectorizer.transform([processed_input])
        predicted_response = model.predict(input_vector)[0]
        return predicted_response
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()  # Strip whitespace
    if not user_input:
        return jsonify({'error': 'Empty input'}), 400
    try:
        bot_response = get_response(user_input, vectorizer, random_forest_model, df)
    except ValueError as ve:
        return jsonify({'error': f"Model error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

    return jsonify({'user_message': user_input, 'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)