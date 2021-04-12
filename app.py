# ADD THE LIBRARIES YOU'LL NEED
import re
import os
import string
import pandas as pd 
import pickle
from collections import Counter
from string import punctuation
from flask import Flask, request, jsonify, render_template

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
nltk.download('punkt')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#import pad_sequences
import keras
#from tensorflow import keras
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional, GRU


PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
             "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
             'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 
             'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
             'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
             'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
             'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
             'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now',
             'd', 'll', 'm', 'o', 're', 've', 'y']


def convert_to_lower(text):
  text = text.str.lower()
  return text

def remove_stopwords(text):
  text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
  return text

def preprocess_data(review):
  review = convert_to_lower(review)
  review = remove_stopwords(review)
  return review


with open('tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)
  
print("tokenizer",tokenizer)

model = keras.models.load_model('pre-trained-rnn.h5')
# print(model)



@app.route('/')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'nlp.jpg')
    return render_template("index.html", user_image = full_filename)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [(x) for x in request.form.values()]
    features = features[0]
    features = pd.Series(features)
    features = preprocess_data(features)
    features = features.str.replace("didnt", "didn't")
	features = features.str.replace("doesnt", "doesn't")
	features = features.str.replace("couldnt", "couldn't")
	features = features.str.replace("dont", "don't")
	features = features.str.replace("havent", "haven't")
	features = features.str.replace("wouldnt", "wouldn't")
	features = features.str.replace("wont", "won't")
	features = features.str.replace("shouldnt", "shouldn't")
	
    features = tokenizer.texts_to_sequences(features)
    features = pad_sequences(features, maxlen=40)
    
    y_pred = model.predict(features)
    class1 = y_pred[0][0]
    class2 = y_pred[0][1]
    class3 = y_pred[0][2]
    class4 = y_pred[0][3]
    class5 = y_pred[0][4]

    for y in y_pred:
        t = y.tolist()
        value = t.index(max(t)) + 1

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'nlp.jpg')
    return render_template('index.html',user_image = full_filename ,prediction_text='{}'.format(value),
    class1='{}'.format(class1),class2='{}'.format(class2),class3='{}'.format(class3),
    class4='{}'.format(class4),class5='{}'.format(class5))


if __name__ == "__main__":
    app.run(debug=True)
