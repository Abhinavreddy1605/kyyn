from flask import Flask, request, render_template
import joblib
from transformers import pipeline
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Initialize Flask app
app = Flask(__name__)

# Load the Random Forest model, vectorizer, and scaler
rf_model = joblib.load('models/random_forest_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')

# Load BERT model (DistilBERT) for sentiment analysis
bert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Store the inputs from users
user_inputs = []
sentiment_counts = {'positive': 0, 'negative': 0}

# Preprocessing function for Random Forest model
def preprocess_text_rf(text, vectorizer, scaler):
    # Text cleaning
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)       # Remove mentions and hashtags
    text = re.sub(r'\W+|\d+', ' ', text)        # Remove special characters and numbers
    text = text.strip()

    # Tokenization, stopword removal, and lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) if word.lower() not in stop_words]
    clean_text = ' '.join(tokens)

    # TF-IDF transformation and scaling for Random Forest model
    tfidf_vector = vectorizer.transform([clean_text])
    scaled_vector = scaler.transform(tfidf_vector)
    return scaled_vector

# Function to get BERT's sentiment prediction
def get_bert_prediction(text):
    result = bert_classifier(text)
    return result[0]['label']

# Function to get Random Forest's sentiment prediction
def get_rf_prediction(text):
    rf_input = preprocess_text_rf(text, vectorizer, scaler)  # Preprocess text for RF model
    rf_prediction = rf_model.predict(rf_input)[0]  # Get RF prediction
    return rf_prediction

# Function to combine predictions and return the best one
def combine_predictions(bert_pred, rf_pred):
    # Example logic: Prioritize BERT's prediction over Random Forest prediction.
    return bert_pred

# Function to extract trending topics (frequent words)
def extract_trending_topics(text_data):
    all_tokens = []
    for text in text_data:
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)       # Remove mentions and hashtags
        text = re.sub(r'\W+|\d+', ' ', text)        # Remove special characters and numbers
        text = text.strip()

        # Tokenization, stopword removal, and lemmatization
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) if word.lower() not in stop_words]
        all_tokens.extend(tokens)

    # Count word frequencies
    word_counts = Counter(all_tokens)
    most_common_words = word_counts.most_common(10)
    return most_common_words

# Function to generate sentiment count graph
def generate_sentiment_graph():
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')

    # Convert plot to PNG image
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode('utf8')  # Encode the image as base64
    return graph_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']  # Get input text from form
    
    # Get prediction from both models
    bert_prediction = get_bert_prediction(text)
    rf_prediction = get_rf_prediction(text)
    
    # Combine predictions and use only the best one
    final_prediction = combine_predictions(bert_prediction, rf_prediction)
    
    # Update sentiment counts
    if final_prediction == 'POSITIVE':
        sentiment_counts['positive'] += 1
    elif final_prediction == 'NEGATIVE':
        sentiment_counts['negative'] += 1
    
    # Store the user input
    user_inputs.append(text)
    
    # Extract trending topics from the stored inputs
    trending_topics = extract_trending_topics(user_inputs)
    
    # Generate sentiment count graph
    sentiment_graph = generate_sentiment_graph()
    
    # Render result using the best prediction (final_prediction) and trending topics
    return render_template('index.html', prediction=final_prediction, text=text, trending_topics=trending_topics, sentiment_graph=sentiment_graph)

if __name__ == '__main__':
    app.run(debug=True)
