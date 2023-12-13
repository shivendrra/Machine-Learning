import os
import pandas as pd

os.chdir("D:/Machine Learning/NLP")

dataFrame = pd.read_csv("CSV Files/train.csv")     # importing data

y_row = dataFrame['label']      # get the dependent features
x_row = dataFrame.drop('label', axis=1)      # get all the independent features of data by dropping the label

dataFrame = dataFrame.dropna()      # drop the NaN values of the database
messages = dataFrame.copy()      # copying the dataframe
messages.reset_index(inplace=True)      # reset the index of the remaining database

# clearing and preprocessing the data
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
corpus = []     # creating an empty corpus for storing the paras

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)

# training the data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

cv = CountVectorizer(max_features=5800, ngram_range=(1, 3))
x = cv.fit_transform(corpus).toarray()

y = messages['label']

# divide the dataset for training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

feature_names = cv.get_feature_names_out()[:20]
print(feature_names)