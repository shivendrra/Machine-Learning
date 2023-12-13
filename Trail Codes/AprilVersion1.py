#  youtube copied version

import nltk
from spellchecker import SpellChecker
import urllib
import bs4 as bs
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
import random
from sklearn.metrics.pairwise import cosine_similarity
import random
import string 
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import pyttsx3 
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer

page1=requests.get('https://www.timeanddate.com/weather/india')

def temp(topic):
    
    page = page1
    soup = BeautifulSoup(page.content,'html.parser')

    data = soup.find(class_ = 'zebra fw tb-wt zebra va-m')

    tags = data('a')
    city = [tag.contents[0] for tag in tags]
    tags2 = data.find_all(class_ = 'rbi')
    temp = [tag.contents[0] for tag in tags2]

    indian_weather = pd.DataFrame(
    {
        'City':city,
        'Temperature':temp
    }
    )
    
    df = indian_weather[indian_weather['City'].str.contains(topic.title())] 
    
    return (df['Temperature'])

def wiki_data(topic):
    
    topic=topic.title()
    topic=topic.replace(' ', '_',1)
    url1="https://en.wikipedia.org/wiki/"
    url=url1+topic

    source = urllib.request.urlopen(url).read()

    # Parsing the data/ creating BeautifulSoup object
    soup = bs.BeautifulSoup(source,'lxml')

    # Fetching the data
    text = ""
    for paragraph in soup.find_all('p'):
        text += paragraph.text

    import re
    # Preprocessing the data
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    
    
    return (text)
    
def rem_special(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return(text.translate(remove_punct_dict))
sample_text="I am sorry! I don't understand you."
rem_special(sample_text)

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

def stemmer(text):
    words = word_tokenize(text) 
    for w in words:
        text=text.replace(w,PorterStemmer().stem(w))
    return text

stemmer("He is Eating. He played yesterday. He will be going tomorrow.")

lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

sample_text="rocks corpora better" #default noun
LemTokens(nltk.word_tokenize(sample_text))

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

remove_stopwords("This is a sample sentence, showing off the stop words filtration.")

import spacy 
spacy_df=[]
spacy_df1=[]
df_spacy_nltk=pd.DataFrame()
nlp = spacy.load('en_core_web_sm') 
  
# Process whole documents 
sample_text = ("The heavens are above. The moral code of conduct is above the civil code of conduct") 
doc = nlp(sample_text) 
  
# Token and Tag 
for token in doc:
    spacy_df.append(token.pos_)
    spacy_df1.append(token)


df_spacy_nltk['origional']=spacy_df1
df_spacy_nltk['spacy']=spacy_df

spell = SpellChecker()
def spelling(text):
    splits = sample_text.split()
    for split in splits:
        text=text.replace(split,spell.correction(split))
        
    return (text)