import os
import pandas as pd
import numpy as np

os.chdir('D:/Machine Learning/Machine Learning Git-repo/Spam-Ham detection/')
df = pd.read_csv('mail_data.csv')
data = df['Message'].values.astype('U')
data = np.array(data)

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

lm = WordNetLemmatizer()

processed_corpus = []
for i in range(len(data)):
  sentence = data[i]
  tokens = nltk.sent_tokenize(sentence)
  for j in range(len(tokens)):
    review = re.sub('[^a-zA-Z]', ' ', tokens[j])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    processed_corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_feat = cv.fit_transform(processed_corpus).toarray()
x_feat = pd.DataFrame(x_feat)
x_feat = np.array(x_feat.iloc[:5500])

y_feat = pd.get_dummies(df['Category'])
y_feat = y_feat.iloc[:5500]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_feat, y_feat, test_size=0.15, random_state=0)
y_train = y_train['spam'].values.flatten()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

from sklearn.naive_bayes import MultinomialNB

spam_detection = MultinomialNB().fit(x_train, y_train)
y_pred = spam_detection.predict(x_test)
y_test = y_test['spam'].values.flatten()

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)