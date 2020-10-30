from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from nltk.stem import SnowballStemmer
from stop_words import get_stop_words
import os, re, json, pickle
from nltk.corpus import stopwords
from unidecode import unidecode
import pandas as pd

FR = SnowballStemmer('french')
MY_STOP_WORD_LIST = get_stop_words('french')
FINAL_STOPWORDS_LIST = stopwords.words('french')

S_W = list(set(FINAL_STOPWORDS_LIST + MY_STOP_WORD_LIST))
S_W = [elem.lower() for elem in S_W]

CLS = pickle.load(open("./data/model.pkl", "rb"))
LOADED_VEC = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("./data/base.pkl", "rb")))

VECTORIZER = TfidfVectorizer()


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')


@app.route("/entrainement", methods=['POST'])
def entrainement():
    data = pd.read_csv('./data/news.csv')
    score = doTraining(data)
    return json.dumps({'response': 'Le training est fini le score est de {} sur un dataset de {} documents en 50/50'.format(score['accuracy'], score['size'])})


@app.route("/prediction", methods=['POST'])
def prediction():
    user_text = request.form.get('input_text')
    res = getPrediction(user_text)
    return json.dumps({'response': 'Le article est une {} avec {} % de probabilité'.format(res['valeurText'], res['proba'])})


def getPrediction(user_text):
    transformer = TfidfTransformer()
    user = transformer.fit_transform(LOADED_VEC.fit_transform([nettoyage(user_text)]))
    if CLS.predict(user)[0].astype(str) == '1':
        valeurText = "Fake News"
    else:
        valeurText = "Vrai News"
    proba = round(CLS.predict_proba(user).max(), 2) * 100
    return {
        "valeurText": valeurText,
        "proba": proba
    }


def doTraining(data):
    x = vectorisation(data['title'].apply(nettoyage))
    pickle.dump(VECTORIZER.vocabulary_, open("./data/base.pkl", "wb"))

    y = data['label']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    cls = LogisticRegression(max_iter=300).fit(x_train, y_train)
    pickle.dump(cls, open("./data/model.pkl", "wb"))
    CLS = pickle.load(open("./data/model.pkl", "rb"))
    LOADED_VEC = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("./data/base.pkl", "rb")))
    return {
        "accuracy": round(cls.score(x_val, y_val), 2) * 100,
        "size": len(data['title'])
    }


def nettoyage(string):
    l = []
    string = unidecode(string.lower())
    string = " ".join(re.findall("[a-zA-Z]+", string))

    for word in string.split():
        if word in S_W:
            continue
        else:
            l.append(FR.stem(word))
    return ' '.join(l)

def vectorisation(text):
    VECTORIZER.fit(text)
    return VECTORIZER.transform(text)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 80)