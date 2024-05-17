import math
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

def build_vocabulary(corpus):
  vocabulary = set()
  for document in corpus:
    tokens = word_tokenize(document)
    vocabulary.update(tokens)
  return sorted(vocabulary)

def term_frequency(document, vocabulary):
  tokens = word_tokenize(document)
  tf = [0] * len(vocabulary)
  for token in tokens:
    if token in vocabulary:
      index = vocabulary.index(token)
      tf[index] += 1
  tf = [freq / len(tokens) for freq in tf]
  return tf

def inverse_document_frequency(corpus, vocabulary):
  idf = [0] * len(vocabulary)
  num_documents = len(corpus)
  for i, word in enumerate(vocabulary):
    count = sum(1 for document in corpus if word in word_tokenize(document))
    idf[i] = math.log((num_documents + 1) / (count + 1)) + 1
  return idf

def tf_idf(corpus):
  vocabulary = build_vocabulary(corpus)
  idf = inverse_document_frequency(corpus, vocabulary)
  tf_idf_vectors = []
  for document in corpus:
    tf = term_frequency(document, vocabulary)
    tf_idf_vector = [tf[i] * idf[i] for i in range(len(vocabulary))]
    tf_idf_vectors.append(tf_idf_vector)
  return tf_idf_vectors, vocabulary

text = """
Implementations of the bag-of-words model might involve using frequencies of words in a document to represent its contents.
The frequencies can be "normalized" by the inverse of document frequency, or tf-idf. Additionally, for the specific purpose of classification,
supervised alternatives have been developed to account for the class label of a document.[4] Lastly, binary (presence/absence or 1/0)
weighting is used in place of frequencies for some problems (e.g., this option is implemented in the WEKA machine learning software system).
"""

vectors, vocabulary = tf_idf(sent_tokenize(text))

print("Vocabulary:", vocabulary)
print("TF-IDF Vectors:")
for vector in vectors:
  print(vector)