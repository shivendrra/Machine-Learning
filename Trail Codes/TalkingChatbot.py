### Training Part ###
import speech_recognition as sr
import pyttsx3
import numpy as np
import pandas as pd

# crerating objects
recon = sr.Recognizer()
engine = pyttsx3.init()

# customization of voice and its properties
sound = engine.getProperty('voices')
engine.setProperty('voice', sound[1].id)
engine.setProperty('rate', 160)

# preprocess the training data
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

ps =PorterStemmer()
lm = WordNetLemmatizer()
data = pd.read_csv('d:/Machine Learning/NLP/CSV Files/chatdataset.csv')

# separating the feature
input_para = data['input'].values.astype("U")
output_para = data['output'].values.astype("U")

def preprocess(inp):
    corpus = []
    for i in range(len(input_para)):
        sentence = input_para[i]
        sent_token = nltk.sent_tokenize(sentence)
        for i in range(len(sent_token)):
            text = re.sub('[^a-zA-Z]', ' ', sent_token[i])
            text = text.lower()
            text = text.split()
            text = [lm.lemmatize(word) for word in text if not word in stopwords.words('english')]
            text = ' '.join(text)
        corpus.append(text)
    return corpus            

newCorp = preprocess(input_para) + preprocess(output_para)  # adding both input and output data together

# applying word2vec, TfIdf or bag of words
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

tv = TfidfVectorizer()
tt = TfidfTransformer()
x_feat = tv.fit_transform(newCorp).toarray()

# train the program using LSTM and RNN

### Input Part ###
# input of voice
with sr.Microphone() as source:
    print("Listening: ")
    audio = recon.listen(source)
    input_text = recon.recognize_google(audio)

# preprocess the input data
in_corpus = []
for i in range(len(input_text)):
    text = re.sub('[^a-zA-Z]', ' ', input_text[i])
    text = text.lower()
    text = text.split()
    text = [lm.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
in_corpus.append(text)
in_feat = tv.fit_transform(in_corpus).toarray()  # applying tfIdf to input data


### Output Part ###
# try to reply based on training data
