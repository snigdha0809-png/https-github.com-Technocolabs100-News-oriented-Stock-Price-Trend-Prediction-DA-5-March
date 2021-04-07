# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:38:01 2021

@author: Snigdha Nandy
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

data= pd.read_csv(r"D:/Combined_News_DJIA.csv")

data.head()
#data.info()


data["Top"] = data["Top1"] + data["Top2"]+data["Top3"]+data["Top4"]+data["Top5"]+\
    data["Top6"]+data["Top7"]+data["Top8"]+data["Top9"]+data["Top10"]+data["Top11"]+\
        data["Top12"]+data["Top13"]+data["Top14"]+data["Top15"]+data["Top16"]+data["Top17"]+\
            data["Top18"]+data["Top19"]+data["Top20"]+data["Top21"]+data["Top22"]+data["Top23"]+\
                data["Top24"]+data["Top25"]
            
data.drop(["Top1","Top2","Top3","Top4","Top5","Top6","Top7","Top8","Top9","Top10","Top11",\
        "Top12","Top13","Top14","Top15","Top16","Top17","Top18","Top19","Top20","Top21","Top22","Top23"\
                ,"Top24","Top25"],axis=1,inplace=True)

data.isnull().sum()

data = data.replace(np.nan, ' ', regex=True)
##split the data
train= data[data['Date']< '20150101' ]
test=  data[data['Date']> '20141231']
train = train.drop(['Date'], axis=1)
test = test.drop(['Date'], axis=1)

###################3logistic regrassion####################
tfidf_vectorizer= TfidfVectorizer()
## fit the data with training data
tfidf_vectorizer.fit_transform(train.Top)

train_idf=tfidf_vectorizer.transform(train.Top)
test_idf=tfidf_vectorizer.transform(test.Top)

model_logrig=LogisticRegression()
##fit the model in training data
model_logrig.fit(train_idf,train.Label)
##predict the model on training data
predict_train=model_logrig.predict(train_idf)
##predict the model on test data
predict_test=model_logrig.predict(test_idf)

matrix = confusion_matrix(test['Label'], predict_test)
print(matrix)
score = accuracy_score(test['Label'], predict_test)
print(score)
report = classification_report(test['Label'], predict_test)
print(report)

#############Random forest Classifier#######################
### Bag of word
countvec = CountVectorizer()
## fit the data with training data
countvec.fit_transform(train.Top)
train_countvec=countvec.transform(train.Top)
test_countvec=countvec.transform(test.Top)

model_ranforcls=RandomForestClassifier()
##fit the model in training data
model_ranforcls.fit(train_idf,train.Label)
##predict the model on training data
predict_train=model_ranforcls.predict(train_idf)
##predict the model on test data
predict_test=model_ranforcls.predict(test_idf)
##f1 score on train data
f1_score(y_true=train.Label, y_pred=predict_train)
## f1 score on test data
f1_score(y_true=test.Label, y_pred=predict_test)

matrix = confusion_matrix(test['Label'], predict_test)
print(matrix)
score = accuracy_score(test['Label'], predict_test)
print(score)
report = classification_report(test['Label'], predict_test)
print(report)


