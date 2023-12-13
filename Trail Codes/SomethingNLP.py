import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
import re
import speech_recognition as sr
import nltk 
import pyttsx3

# creating objects
r = sr.Recognizer()
engine = pyttsx3.init()

# taking the input
with sr.Microphone() as source:
    print('Say: ')
    audio = r.listen(source)
    input_text = r.recognize_google(audio)

# Preprocessing the input
input_text = input_text.lower()
fresh_text = re.sub(r'\[[0-9]*\]', ' ', input_text)
fresh_text = re.sub(r'\s+', ' ',fresh_text)
fresh_text = re.sub(r'\d+', ' ',fresh_text)
fresh_text = re.sub(r'\s+', ' ',fresh_text)

# Preparing the dataset
sentence = nltk.sent_tokenize(fresh_text)
sentence = [nltk.word_tokenize(s) for s in sentence]

# Removing the stopwords
for i in range(len(sentence)):
    sentence[i] = [word for word in sentence[i] if word not in stopwords.words('english')]

print(sentence)

# Training the model
model = word2vec.Word2Vec(sentence, min_count=1)

# speaking it out loud
engine.say(input_text)

# changing the voice
sound = engine.getProperty('voices')
engine.setProperty('voice', sound[0].id)
 
engine.runAndWait()


# In military terminology, a missile is a guided airborne ranged weapon capable of self-propelled flight usually by a jet engine or rocket motor. Missiles are thus also called guided missiles or guided rockets (when in rocket form). Missiles have five system components: targeting, guidance system, flight system, engine and warhead. Missiles come in types adapted for different purposes: surface-to-surface and air-to-surface missiles (ballistic, cruise, anti-ship, anti-tank, etc.), surface-to-air missiles (and anti-ballistic), air-to-air missiles, and anti-satellite weapons.
