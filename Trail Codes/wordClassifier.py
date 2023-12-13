# working

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

file_in = open("australia.txt", "r")
sen = file_in.readlines()
count = 0
sent_new = ""
for line in sen:
    count += 1
    sent_new = sent_new + line.strip() + "\n"
print(sent_new)
sentence = nltk.sent_tokenize(sent_new)

corpus = []
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)
# converting string array to stemmed paragraph
def convert(string):
    new = ""
    for x in string:
        new = new + x + '\n'
    return new
new_corpus = convert(corpus)

file_out = open("output.txt", "w")
file_out.write(new_corpus)
file_out.close()
file_in.close()

print("done!!")