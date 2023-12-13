# still working on this

import speech_recognition as sr
import pyttsx3

# configuration of voice and properties
recon = sr.Recognizer()
engine = pyttsx3.init()

sound = engine.getProperty('voices')
engine.setProperty('voice', sound[1].id)
engine.setProperty('rate', 160)

# speaking it out loud
startSen = "Hello, my name is April, I'm your AI friend, With which name should I call you?"
engine.say(startSen)

engine.runAndWait()

with sr.Microphone() as source:
    print("Listening: ")
    audio = recon.listen(source)
    input_text = recon.recognize_google(audio)

# preprocess the text
import re
import nltk 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

input_text = input_text.lower()
sent_token = nltk.sent_tokenize(input_text)
for i in range(0, len(sent_token)):
    review = re.sub('[^a-zA-Z]', ' ', sent_token[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)


def talk():
    print("talk func")

def listen():
    print("listening")