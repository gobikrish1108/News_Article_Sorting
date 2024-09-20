from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model, tokenizer, and label encoder
model = load_model('text_classifier_model4.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Function to preprocess input text and predict the category
def preprocess_and_predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=500)
    pred = model.predict(padded)
    category = label_encoder.inverse_transform([np.argmax(pred)])
    return category[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    if request.method == 'POST':
        user_input = request.form['text_input']
        if user_input:
            category = preprocess_and_predict(user_input)
    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
