"""
  this code is replica of India In Pixels' code where he tried to explain
  national anthems of every country using ml
  
  - video link: https://youtu.be/a-AqvPtjjts?si=ype1L6GjxLFflXzg
  - dataset: https://www.kaggle.com/datasets/amankrpandey1/text-files-related-to-national-anthem-clustering?select=national_anthems.csv
"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import pandas as pd
data = pd.read_csv('csv/national_anthems.csv')
para = data['Anthem'].values.astype("U")

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

lm = WordNetLemmatizer()

processed_corpus = []
for i in range(len(para)):
  sentence = para[i]
  tokens = nltk.sent_tokenize(sentence)
  for j in range(len(tokens)):
    review = re.sub('[^a-zA-Z]', ' ', tokens[j])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
  processed_corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
vect = tf.fit_transform(processed_corpus).toarray()

from sklearn.cluster import KMeans

k = KMeans(n_clusters=5, random_state=40)
kmeans = k.fit(vect)
prd = kmeans.labels_
clusters = kmeans.cluster_centers_

data['clusters'] = kmeans.labels_
print(data)

anthems = pd.DataFrame(data)
anthems.to_csv('csv/outputs.csv')