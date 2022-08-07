import json
import pickle
import re
from string import punctuation

import nltk
from nltk.util import everygrams
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.util import everygrams

nltk.data.path.append("/tmp")
nltk.download("punkt", download_dir = "/tmp")
nltk.download("stopwords", download_dir = "/tmp")
nltk.download('wordnet', download_dir = "/tmp")


model_file = open('sa_classifier.pickle', 'rb')
model = pickle.load(model_file)


stopwords_eng = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def bag_of_words(words):
    bag = {}
    for w in words:
        bag[w] = bag.get(w,0)+1
    return bag

def is_useful_word(word):
    return (word not in stopwords_eng) and (word not in punctuation)

def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

def extract_features(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if is_useful_word(w)]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w != "" and is_useful_word(w)]
    return [str('_'.join(ngram)) for ngram in list(everygrams(words, max_len=3))]

def get_sentiment(review):
    words = extract_features(review)
    words = bag_of_words(words)
    return model.classify(words)

