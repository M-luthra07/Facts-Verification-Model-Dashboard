from flask import Flask, render_template, request
import re
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

app = Flask(__name__)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = tf.keras.models.load_model('model/bert_lstm_model.keras')

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Prepare text for prediction
def prepare_input(text, max_length=128):
    text = clean_text(text)
    tokens = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return tokens['input_ids'], tokens['attention_mask']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['news_input']
        input_ids, attention_mask = prepare_input(input_text)
        prediction = model.predict([input_ids, attention_mask])
        label = "Fake News" if np.argmax(prediction) == 1 else "Real News"
        confidence = float(np.max(prediction)) * 100

        return render_template('predict.html',
                               input_text=input_text,
                               prediction=label,
                               confidence=round(confidence, 2))


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
