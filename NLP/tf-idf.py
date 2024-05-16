import math
import string

def tokenize(text):
  """
  Tokenizes the input text by converting to lowercase, removing punctuation, and splitting on whitespace.
  Args:
    text (str): The input text.
  Returns:
    list: List of word tokens.
  """
  text = text.lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  tokens = text.split()
  return tokens

def build_vocabulary(corpus):
  """
  Builds a vocabulary from a list of documents.
  Args:
    corpus (list of str): List of documents.
  Returns:
    list: List of unique words in the corpus.
  """
  vocabulary = set()
  for document in corpus:
    tokens = tokenize(document)
    vocabulary.update(tokens)
  return sorted(vocabulary)

def term_frequency(document, vocabulary):
  """
  Calculates the term frequency for each word in the vocabulary for a given document.
  Args:
    document (str): The document.
    vocabulary (list): The vocabulary list.
  Returns:
    list: Term frequency vector for the document.
  """
  tokens = tokenize(document)
  tf = [0] * len(vocabulary)
  for token in tokens:
    if token in vocabulary:
      index = vocabulary.index(token)
      tf[index] += 1
  tf = [freq / len(tokens) for freq in tf]
  return tf

def inverse_document_frequency(corpus, vocabulary):
  """
  Calculates the inverse document frequency for each word in the vocabulary.
  Args:
    corpus (list of str): List of documents.
    vocabulary (list): The vocabulary list.
  Returns:
    list: Inverse document frequency vector.
  """
  idf = [0] * len(vocabulary)
  num_documents = len(corpus)
  for i, word in enumerate(vocabulary):
    count = sum(1 for document in corpus if word in tokenize(document))
    idf[i] = math.log((num_documents + 1) / (count + 1)) + 1
  return idf

def tf_idf(corpus):
  """
  Converts a corpus of documents into a TF-IDF representation.
  Args:
    corpus (list of str): List of documents.
  Returns:
    list of list: TF-IDF representation of the corpus.
    list: Vocabulary list.
  """
  vocabulary = build_vocabulary(corpus)
  idf = inverse_document_frequency(corpus, vocabulary)
  tf_idf_vectors = []
  for document in corpus:
    tf = term_frequency(document, vocabulary)
    tf_idf_vector = [tf[i] * idf[i] for i in range(len(vocabulary))]
    tf_idf_vectors.append(tf_idf_vector)
  return tf_idf_vectors, vocabulary

corpus = [ "The quick brown fox jumps over the lazy dog.", "The dog barks at the fox.", "Quick brown fox and quick dog."
]

tf_idf_vectors, vocabulary = tf_idf(corpus)

print("Vocabulary:", vocabulary)
print("TF-IDF Vectors:")
for vector in tf_idf_vectors:
  print(vector)