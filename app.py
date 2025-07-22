from flask import Flask, render_template, request
import re
import numpy as np
import tensorflow as tf
import os
from transformers import BertTokenizer

app = Flask(__name__)
@register_keras_serializable()
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model_name='bert-base-uncased', **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(bert_model_name)
        self.bert.trainable = True

    def call(self, inputs, **kwargs):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL_PATH = os.getenv('MODEL_PATH', 'model/bert_lstm_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

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

port = int(os.environ.get("PORT", 10000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
