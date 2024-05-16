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

def vectorize(document, vocabulary):
  """
  Creates a vector representation of a document based on a given vocabulary.
  Args:
    document (str): The document to vectorize.
    vocabulary (list): The vocabulary list.
  Returns:
    list: The document vector.
  """
  tokens = tokenize(document)
  vector = [0] * len(vocabulary)
  for token in tokens:
    if token in vocabulary:
      index = vocabulary.index(token)
      vector[index] += 1
  return vector

def bag_of_words(corpus):
  """
  Converts a corpus of documents into a bag-of-words representation.
  Args:
    corpus (list of str): List of documents.
  Returns:
    list of list: The bag-of-words representation of the corpus.
  """
  vocabulary = build_vocabulary(corpus)
  vectors = [vectorize(document, vocabulary) for document in corpus]
  return vectors, vocabulary

corpus = ["The quick brown fox jumps over the lazy dog.", "The dog barks at the fox.", "Quick brown fox and quick dog."]

vectors, vocabulary = bag_of_words(corpus)

print("Vocabulary:", vocabulary)
print("Vectors:")
for vector in vectors:
  print(vector)