import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
from xgboost import XGBClassifier
import math
from sklearn.linear_model import LogisticRegression
warnings.simplefilter("ignore")
import nltk
# nltk.download('all')
import pandas as pd

# df = pd.read_csv(io.BytesIO(uploaded['dis_sym_dataset_comb.csv']))
df = pd.read_csv("dis_sym_dataset_comb.csv")
# df1 = pd.read_csv(io.BytesIO(uploaded1['dis_sym_dataset_norm.csv']))
df1 = pd.read_csv("dis_sym_dataset_norm.csv")

# creation of features and label for training the models
X = df.iloc[:, 1:]
Y = df.iloc[:, 0:1]


def synonym(term):

    
    # utlities for pre-processing
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    splitter = RegexpTokenizer(r'\w+')

    # Taking symptoms from user as input 
    user_symptoms = term.lower().split(',')
    # Preprocessing the input symptoms
    processed_user_symptoms=[]
    for sym in user_symptoms:
        sym=sym.strip()
        sym=sym.replace('-',' ')
        sym=sym.replace("'",'')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)


        # Taking each user symptom and finding all its synonyms and appending it to the pre-processed symptom string
    user_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym)+1):
            for subset in combinations(user_sym, comb):
                subset=' '.join(subset)
                subset = synonyms(subset) 
                str_sym.update(subset)
        str_sym.add(' '.join(user_sym))
        user_symptoms.append(' '.join(str_sym).replace('_',' '))
        print("After query expansion done by using the symptoms entered")
    return(user_symptoms)

def synonyms(term):
    
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container=soup.find('section', {'class': 'MainContentContainer'}) 
        row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms+=syn.lemma_names()
    return set(synonyms)
