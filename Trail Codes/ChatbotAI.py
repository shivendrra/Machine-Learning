### Collecting and Pre-processing ###
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

# creating objects
ps =PorterStemmer()
lm = WordNetLemmatizer()

data = pd.read_csv('d:/Machine Learning/NLP/CSV Files/chatdataset.csv')

# separating the feature
input_para = data['input'].values.astype("U")
output_para = data['output'].values.astype("U")


### Training the data ###

### Validating ###

### Input Processing ###

### Generating Output ###

### Giving Output ###