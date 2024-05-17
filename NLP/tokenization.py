import re

def sent_tokenize(corpus):
  pattern = r'(?<=[.!?])\s+'
  sent = re.split(pattern, corpus.strip())
  sent = [i.replace('\n', ' ') for i in sent]
  return sent

def word_tokenize(corpus):
  pattern = r'\b\w+(?:-\w+)*\b|``|\'\'|\[\d+\]|\(|\)|\d+/0|e\.g\.|\.|,|;|:|!'
  corpus = re.findall(pattern, corpus)
  return corpus

text = """
Implementations of the bag-of-words model might involve using frequencies of words in a document to represent its contents.
The frequencies can be "normalized" by the inverse of document frequency, or tf-idf. Additionally, for the specific purpose of classification,
supervised alternatives have been developed to account for the class label of a document.[4] Lastly, binary (presence/absence or 1/0)
weighting is used in place of frequencies for some problems (e.g., this option is implemented in the WEKA machine learning software system).
"""

sent = sent_tokenize(text)
print(sent)

words = word_tokenize(text)