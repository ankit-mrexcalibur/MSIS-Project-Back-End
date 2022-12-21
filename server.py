
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')


def get_relevant_columns(category_count_series , THRESHOLD):
  # returns only the columns with unique count > THRESHOLD
  res = []
  for index_val in category_count_series.iteritems():
    if index_val[1] >= THRESHOLD:
      res.append(index_val[0])
    else:
      break
  return res


def remove_punctuation(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text


lemmatizer = WordNetLemmatizer()


def use_lemmatizer(text):
    ret_text = ""
    use_text = ""
    for char in text:
        if not char.isdigit():
            use_text += char

    words = word_tokenize(use_text)
    res_tokens = []
    for word in words:
        # ret_text += lemmatizer.lemmatize(word) + " "
        res_tokens.append(lemmatizer.lemmatize(word).lower())

    return res_tokens


def remove_stop_words(df):
    stop_words_ = stopwords.words('english')
    # df['transcription'] = df['transcription'].apply(stop_words_func)
    df['transcription'] = df['transcription'].apply(
        lambda x: ' '.join([word for word in x if word not in (stop_words_)]))


def PreProcess(input_text):
    df = pd.DataFrame([input_text], columns=['transcription'])
    df['transcription'] = df['transcription'].apply(remove_punctuation)
    df['transcription'] = df['transcription'].apply(use_lemmatizer)
    remove_stop_words(df)
    vectorizer= pickle.load(open('tfidf_vectorizer.pk', 'rb'))
    tfIdfMat = vectorizer.transform(df['transcription'].tolist())
    

    from sklearn.decomposition import PCA

    pca = pickle.load(open('pca_obj.pk','rb'))
    tfIdfMat_pca = pca.transform(tfIdfMat.toarray())
    print("***************************")
    print(tfIdfMat_pca.shape)

    num_to_label = pickle.load(open('num_to_label.sav', 'rb'))
    
    X_test = tfIdfMat_pca

    #model file
    filename = 'svm_model.pk'
    loaded_model = pickle.load(open(filename, 'rb'))
    print("**1")
    y_pred = loaded_model.predict(X_test)
    print("**2")
    print(y_pred)
    return jsonify(y_pred[0])

from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])



@app.route('/query')
def results():
    fileData = request.args.get('fileData')
    print(fileData)

    return PreProcess(fileData)

app.run(debug=True, port=5000)








