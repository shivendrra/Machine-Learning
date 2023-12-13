# this code has some errors or bugs

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
import os

os.chdir('d:/Machine learning/Machine Learning Git-repo/K-means Clustering/National Anthems Classifier/')

df = pd.read_csv('anthems.csv')
documents = df['Anthem'].values.astype("U")

doc_arr = np.array(documents)
ps = PorterStemmer()
corpus = []
for i in range(len(doc_arr)):
    sentence = doc_arr[i]
    sent_token = nltk.sent_tokenize(sentence)
    for i in range(0, len(sent_token)):
        review = re.sub('[^a-zA-Z]', ' ', sent_token[i])
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

# implementing tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x_features = cv.fit_transform(corpus).toarray()

count_df = pd.DataFrame(x_features[:, :190], columns=cv.get_feature_names_out()[:190]).reset_index(drop=True)
print(count_df)

# K-means algorithm
from sklearn.cluster import KMeans
k = 5
model = KMeans(n_clusters=k, random_state=40)
kmeans = model.fit(x_features)
predicted = kmeans.labels_
cluster = kmeans.cluster_centers_
print(cluster)

df['clusters'] = model.labels_[:190]
nat_anth = pd.DataFrame(df)
nat_anth.to_csv('nat_anth.csv')
print(nat_anth)

from nltk.probability import FreqDist
sent_corp = ''
for o in range(len(corpus)):
  sent_corp = sent_corp + ' '.join(corpus[o])
sent_words = nltk.word_tokenize(sent_corp)

from sklearn.feature_extraction.text import CountVectorizer
vp = CountVectorizer()
freq = FreqDist(sent_words)

freq_table = pd.DataFrame(list(count_df.items()), columns=['Word', 'Frequency'])
freq_table.to_csv('freq.csv')