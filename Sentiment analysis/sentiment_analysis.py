import os
os.chdir('D:/Machine Learning/Machine Learning Git-repo/Sentiment analysis')

# importing and setting the data
import pandas as pd
import numpy as np

data = pd.read_csv('IMDB Dataset.csv')

# data preprocessing
data['review'].replace('https?://\S+|www\.\S+'," ",regex=True,inplace=True)
data['review'].replace('<.*?>'," ",regex=True,inplace=True)
data['review'].replace('@\w+'," ",regex=True,inplace=True)
data['review'].replace('#\w+'," ",regex=True,inplace=True)
data['review'].replace("[^\w\s\d]"," ",regex=True,inplace=True)
data['review'].replace(r'( +)'," ",regex=True,inplace=True)
data['review'].replace("[^a-zA-Z]"," ",regex=True,inplace=True)

# data seperation
reviews = data['review'].values.astype('U')
reviews = np.array(reviews)

response = data['sentiment'].values.astype('U')
response = np.array(response)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

lm = WordNetLemmatizer()

corpus = []
for i in range(len(reviews)):
  sen = reviews[i]
  token = nltk.sent_tokenize(sen)
  for j in range(len(token)):
    review = re.sub('[^a-zA-Z]', ' ', token[j])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_feat = cv.fit_transform(corpus).toarray()
x_feat = pd.DataFrame(x_feat)
x_feat = np.array(x_feat.iloc[:5500])

y_feat = pd.get_dummies(data['sentiment'])
y_feat = y_feat.iloc[:5500]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feat, y_feat, test_size=0.15, random_state=0)
y_train = y_train['positive'].values.flatten()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

from sklearn.naive_bayes import MultinomialNB

spam_detection = MultinomialNB().fit(x_train, y_train)
y_pred = spam_detection.predict(x_test)
y_test = y_test['positive'].values.flatten()

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)