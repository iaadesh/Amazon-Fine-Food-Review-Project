from flask import Flask, render_template, request, jsonify
import tensorflow
import keras
import numpy as np
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index(): 
    return render_template("index.html")

# loading model weights
model = keras.models.load_model('best_model_lstm.hdf5')

# loading tokenizer file
a_file = open("tokenizer.pkl", "rb")
tokenizer = pickle.load(a_file)
a_file.close()

# text cleaning function
def text_cleaning(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"<[^<>]+>#", " ", text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    # reading review text
    review_text = request.form['review_text']

    # cleaning review text
    review_text = text_cleaning(review_text)

    # tokenizing review text
    review_text_tokenizer = tokenizer.texts_to_sequences([review_text])
    review_text_pad = pad_sequences(review_text_tokenizer, maxlen = 50, padding = 'post')

    # predicting review text
    prediction = model.predict(review_text_pad)
    if prediction[0][0] >= 0.5:
        predict = "Positive"
    else:
        predict = "Negative"
    
    return render_template("index.html", prediction = predict)

if __name__ == '__main__':
    app.run(debug=True)
