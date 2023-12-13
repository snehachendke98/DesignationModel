import logging
import pickle
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup

# %matplotlib inline

jobs = pd.read_csv('D:/pythonDesignationModel/Model/ListOfDesignations.csv',encoding='latin-1')
jobs['Job_Titles'].apply(lambda x:len(x.split(' '))).sum()
category = jobs['Job_category'].unique()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

jobs['Job_Titles'] = jobs['Job_Titles'].apply(clean_text)

job_titles = jobs['Job_Titles'].to_list()
job_category = jobs['Job_category'].to_list()
X = [x.lower() for x in job_titles]
y = [a.lower() for a in job_category]
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state =308)

mlp = MLPClassifier(hidden_layer_sizes=(999,501,100),max_iter=1001,learning_rate="invscaling",learning_rate_init=0.001)
lr = Pipeline([('vect',CountVectorizer()),
               ('tfidf',TfidfTransformer()),
               ('clf',mlp)])

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(f"Accuracy is :{accuracy_score(y_pred,y_test)}")
pickle_path = "D:/pythonDesignationModel/Model"
with open('count_vectorizer.pkl','wb') as file:
    pickle.dump(lr.named_steps['vect'],file)

with open('tfidf_transformer.pkl','wb') as file:
    pickle.dump(lr.named_steps['tfidf'],file)

with open('mlp_classifier.pkl','wb') as file:
    pickle.dump(lr.named_steps['clf'],file)
