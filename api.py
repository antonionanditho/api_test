# IMPORT LIBRARY FRAMEWORK FOR MACHINE LEARNING
import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import string
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
#from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize


# nltk.download('stopwords')
# nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# IMPORT LIBRARY FOR FLASK
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

app = Flask(__name__)

# SWAGGER UI
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Sentiment Analysis API'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Sentiment Analysis API'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

# parameter Ekstraksi Fitur dan kelas Tokenizer
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

# Label Sentiment
sentiment = ['negative', 'neutral', 'positive']

# cleaning process
punctuations = re.sub(r"[!<_>#:)\.]", "", string.punctuation)

def punct2wspace(text):
    return re.sub(r"[{}]+".format(punctuations), " ", text)

def normalize_wspace(text):
    return re.sub(r"\s+", " ", text)

def casefolding(text):
    return text.lower()
#clean stopwords
stopword = set(stopwords.words('indonesian'))
def clean_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in stopword) # hapus stopword dari kolom text
    return text

stopword_eng = set(stopwords.words('english'))
def clean_stopwords_eng(text):
    text = ' '.join(word for word in text.split() if word not in stopword_eng) # hapus stopword dari kolom text
    return text
def stemmingtokenization(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = word_tokenize(text)
    final = [stemmer.stem(tagged_word) for tagged_word in tokens]
    return " ".join(final)
def preprocess_text(text):
    text = punct2wspace(text)
    text = normalize_wspace(text)
    text = casefolding(text)
    text = stemmingtokenization(text)
    #text = clean_stopwords(text)
    #text = clean_stopwords_eng(text)
    #text = final(text)
    return text
#make function to clean text and make a new "clean_text" column
#import re
#def cleansing(text):
 # text = text.lower()
  #text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
  #return text
#sentiment['clean_text'] = sentiment.Text.apply(preprocess_text)

# LSTM
# # Load model from LSTM
model_file_from_lstm = load_model(r'mymodel.h5')
model_file_from_nn = load_model(r'mymodel_NN.h5')

# Define endpoint LSTM 1
@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    # get the text
    original_text = request.form.get('text1')
    
    # cleaning text
    text = [preprocess_text(str(original_text))]
    # # feature extraction
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=16571)
    # #feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    # # model predict
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    # Define API Response
    json_response = {
        'status_code': 200,
        'description': "Hasil prediksi sentimen menggunakan LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data
# Define endpoint LSTM 2
@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])
def lstm_file():
    # Uploaded file
    file = request.files.getlist('file')[0]

    # import file csv ke Pandas
    df = pd.read_csv(file,encoding='latin-1')

    # Get text from file in list format
    texts = df.Tweet.to.list()

    # Loop list or original text and predict to model
    text_with_sentiment = []
    for original_text in texts:

        # Cleansing
        text = [preprocess_text(original_text)]
        # Feature extraction
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=16571)
        # Inference
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        # Predict "text_clean" to the model. And Insert to list "text_with_sentiment"
        text_with_sentiment.append({
            'text': original_text,
            'sentiment': get_sentiment
        })

    # Define API response
    json_response = {
        'status_code':200,
        'description': "Teks yang sudah diproses",
        'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data

# Define endpoint NN 1
@swag_from("docs/nn.yml", methods=['POST'])
@app.route('/nn', methods=['POST'])
def nn():
    # get the text
    original_text = request.form.get('text1')
    
    # cleaning text
    text = [preprocess_text(str(original_text))]
    # # feature extraction
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=16571)
    # #feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    # # model predict
    prediction = model_file_from_nn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    # Define API Response
    json_response = {
        'status_code': 200,
        'description': "Hasil prediksi sentimen menggunakan LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

# Define endpoint NN 2
@swag_from("docs/nn_file.yml", methods=['POST'])
@app.route('/nn-file', methods=['POST'])
def lstm_file():
    # Uploaded file
    file = request.files.getlist('file')[0]

    # import file csv ke Pandas
    df = pd.read_csv(file,encoding='latin-1')

    # Get text from file in list format
    texts = df.Tweet.to.list()

    # Loop list or original text and predict to model
    text_with_sentiment = []
    for original_text in texts:

        # Cleansing
        text = [preprocess_text(original_text)]
        # Feature extraction
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=16571)
        # Inference
        prediction = model_file_from_nn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        # Predict "text_clean" to the model. And Insert to list "text_with_sentiment"
        text_with_sentiment.append({
            'text': original_text,
            'sentiment': get_sentiment
        })

    # Define API response
    json_response = {
        'status_code':200,
        'description': "Teks yang sudah diproses",
        'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    




