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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import re
from googlesearch import search
import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup

# df = pd.read_csv(io.BytesIO(uploaded['dis_sym_dataset_comb.csv']))
df = pd.read_csv("dis_sym_dataset_comb.csv")
# df1 = pd.read_csv(io.BytesIO(uploaded1['dis_sym_dataset_norm.csv']))
df1 = pd.read_csv("dis_sym_dataset_norm.csv")

X = df1.iloc[:, 1:]
Y = df1.iloc[:, 0:1]

# List of symptoms
dataset_symptoms = list(X.columns)


def part1(term):
        
    # returns the list of synonyms of the input word from thesaurus.com (https://www.thesaurus.com/) and wordnet (https://www.nltk.org/howto/wordnet.html)
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


    # utlities for pre-processing
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    splitter = RegexpTokenizer(r'\w+')

    """**Symptoms initially taken from user.**"""

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
    # query expansion performed by joining synonyms found for each symptoms initially entered
    print("After query expansion done by using the symptoms entered")
    print(user_symptoms)

    # Loop over all the symptoms in dataset and check its similarity score to the synonym string of the user-input 
    # symptoms. If similarity>0.5, add the symptom to the final list
    found_symptoms = set()
    for idx, data_sym in enumerate(dataset_symptoms):
        data_sym_split=data_sym.split()
        for user_sym in user_symptoms:
            count=0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count+=1
            if count/len(data_sym_split)>0.5:
                found_symptoms.add(data_sym)
    found_symptoms = list(found_symptoms)
    # print(found_symptoms)
    return found_symptoms
    # for idx, symp in enumerate(found_symptoms):
    #     return(idx,":",symp)


def part2(term2, found_symptoms):    
    # Show the related symptoms found in the dataset and ask user to select among them
    select_list = term2.split()

    # Find other relevant symptoms from the dataset based on user symptoms based on the highest co-occurance with the
    # ones that is input by the user
    dis_list = set()
    final_symp = []
    counter_list = []
    for idx in select_list:
        symp=found_symptoms[int(idx)]
        final_symp.append(symp)
        dis_list.update(set(df1[df1[symp]==1]['label_dis']))
    
    for dis in dis_list:
        row = df1.loc[df1['label_dis'] == dis].values.tolist()
        row[0].pop(0)
        for idx,val in enumerate(row[0]):
            if val!=0 and dataset_symptoms[idx] not in final_symp:
                counter_list.append(dataset_symptoms[idx])

    # Symptoms that co-occur with the ones selected by user              
    dict_symp = dict(Counter(counter_list))
    dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)   
    print(dict_symp_tup)

    # Iteratively, suggest top co-occuring symptoms to the user and ask to select the ones applicable 
    found_symptoms=[]
    final_symptoms=[]
    count=0
    for tup in dict_symp_tup:
        count+=1
        found_symptoms.append(tup[0])
    final_symptoms=found_symptoms[0:10:]
    part2.var=final_symp
    return(final_symptoms)
        # if count%5==0 or count==len(dict_symp_tup):
        #     print("\nCommon co-occuring symptoms:")
        #     for idx,ele in enumerate(found_symptoms):
        #         print(idx,":",ele)
        #     select_list = input("Do you have have of these symptoms? If Yes, enter the indices (space-separated), 'no' to stop, '-1' to skip:\n").lower().split();
        #     if select_list[0]=='no':
        #         break
        #     if select_list[0]=='-1':
        #         found_symptoms = [] 
        #         continue
        #     for idx in select_list:
        #         final_symp.append(found_symptoms[int(idx)])
        #     found_symptoms = []
def part3(term3, finals):
    
    final_symp2=[]
    terms=term3.split()
    print(terms)
    for i in range(len(terms)):
        x=int(terms[i])
        print(x)
        final_symp2.append(finals[x])
    print(final_symp2)
    for i in range(len(part2.var)):
        final_symp2.append(part2.var[i])
    print(part2.var)
    part3.var=final_symp2
    return(final_symp2)

def diseaseDetail(term):
    diseases=[term]
    ret=term+"\n"
    for dis in diseases:
            # search "disease wilipedia" on google 
        query = dis+' wikipedia'
            # tld="co.in"
            # ,stop=10,pause=0.5
        for sr in search(query): 
                # open wikipedia link
            match=re.search(r'wikipedia',sr)
            filled = 0
            if match:
                wiki = requests.get(sr,verify=False)
                soup = BeautifulSoup(wiki.content, 'html5lib')
                    # Fetch HTML code for 'infobox'
                info_table = soup.find("table", {"class":"infobox"})
                if info_table is not None:
                        # Preprocess contents of infobox
                    for row in info_table.find_all("tr"):
                        data=row.find("th",{"scope":"row"})
                        if data is not None:
                            symptom=str(row.find("td"))
                            symptom = symptom.replace('.','')
                            symptom = symptom.replace(';',',')
                            symptom = symptom.replace('<b>','<b> \n')
                            symptom=re.sub(r'<a.*?>','',symptom) # Remove hyperlink
                            symptom=re.sub(r'</a>','',symptom) # Remove hyperlink
                            symptom=re.sub(r'<[^<]+?>',' ',symptom) # All the tags
                            symptom=re.sub(r'\[.*\]','',symptom) # Remove citation text                     
                            symptom=symptom.replace("&gt",">")
                            ret+=data.get_text()+" - "+symptom+"\n"
                            # print(data.get_text(),"-",symptom)
                            filled = 1
                    if filled:
                        break
    return ret

def part4():
    sample_x = [0 for x in range(0,len(dataset_symptoms))]
    for val in part3.var:
        print(val)
        sample_x[dataset_symptoms.index(val)]=1
    # Predict disease
    print(part3.var)
    lr = LogisticRegression()
    lr = lr.fit(X, Y)
    lr_pred = lr.predict_proba([sample_x])
    # print(lr_pred)

    

    # Predict disease
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=4)
    knn = knn.fit(X, Y)
    knn_pred = knn.predict_proba([sample_x])

    # rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
    # rf = rf.fit(X, Y)
    # rf_pred = rf.predict_proba([sample_x])

    # Multinomial NB Classifier
    mnb = MultinomialNB()
    mnb = mnb.fit(X, Y)
    mnb_pred = mnb.predict_proba([sample_x])

    k = 1
    diseases = list(set(Y['label_dis']))
    diseases.sort()

    # topkrf = rf_pred[0].argsort()[-k:][::-1]
    topkmnb = mnb_pred[0].argsort()[-k:][::-1]
    topkknn = knn_pred[0].argsort()[-k:][::-1]
    topklr = lr_pred[0].argsort()[-k:][::-1]
    # print(topk)

    

    # Take input a disease and return the content of wikipedia's infobox for that specific disease

   

    # print(f"\nTop {k} diseases predicted based on symptoms")
    topk_dict = {}

    my_array = []
    my_arr = []

    # Show top 10 highly probable disease to the user.
    for idx,t in  enumerate(topkmnb):
        # print(idx, t)
        match_sym=set()
        row = df1.loc[df1['label_dis'] == diseases[t]].values.tolist()
        # print(row)
        
        my_array.append(row[0].pop(0))
        # row[0].pop(0)
        

        # PROBABILITY CALCULATE SOCHOOOOOOOOOOOOOOOOOOOOOOOOOOOO
        for idx,val in enumerate(row[0]):
            # print(idx, val)
            if val!=0:
                match_sym.add(dataset_symptoms[idx])
        prob = (len(match_sym.intersection(set(part3.var)))+1)/(len(set(part3.var))+1)
        # prob *= mean(scores)
        topk_dict[t] = prob
        my_arr.append(prob)

    for idx,t in  enumerate(topkknn):
        # print(idx, t)
        match_sym=set()
        row = df1.loc[df1['label_dis'] == diseases[t]].values.tolist()

        my_array.append(row[0].pop(0))
        for idx,val in enumerate(row[0]):
            # print(idx, val)
            if val!=0:
                match_sym.add(dataset_symptoms[idx])
        prob = (len(match_sym.intersection(set(part3.var)))+1)/(len(set(part3.var))+1)
        # prob *= mean(scores)
        # topk_dict[t] = prob
        my_arr.append(prob)
        


    for idx,t in  enumerate(topklr):
        # print(idx, t)
        match_sym=set()
        row = df1.loc[df1['label_dis'] == diseases[t]].values.tolist()
        # print(row)
        
        my_array.append(row[0].pop(0))
        for idx,val in enumerate(row[0]):
            # print(idx, val)
            if val!=0:
                match_sym.add(dataset_symptoms[idx])
        prob = (len(match_sym.intersection(set(part3.var)))+1)/(len(set(part3.var))+1)
        # prob *= mean(scores)
        # topk_dict[t] = prob
        my_arr.append(prob)

    print('Line 328')
    print(my_arr)
    print(my_array)
    #Initialize max with first element of array.    
    max = my_arr[0];    
        
    #Loop through the array    
    for i in range(0, len(my_arr)):    
        #Compare elements of array with max    
        if(my_arr[i] > max):    
            max = my_arr[i];  
    print('Line 339')
    print(my_arr.index(max))
    print()
    return(diseaseDetail(my_array[my_arr.index(max)]))
    

