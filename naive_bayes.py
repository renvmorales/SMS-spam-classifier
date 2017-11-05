#!/usr/bin/python3


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocess import text_process

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle



# read SMS.csv file 
df = pd.read_csv('SMS.csv', encoding = "ISO-8859-1")
print('\nReading csv file ....')



X, Y = shuffle(df['Full_Text'], df['IsSpam'])

msg_train, msg_test, label_train, label_test = train_test_split(X, Y,
	test_size=0.3)



# create a pipeline
pipeline = Pipeline([
	('bow', CountVectorizer(analyzer=text_process)),
	('tfidf', TfidfTransformer()),
	('classifier', MultinomialNB())
	])



def binary_class(x):
	if x=='yes':
		return 1
	else:
		return 0



scores = cross_val_score(pipeline, X, Y.apply(binary_class), 
	cv=10, scoring='f1')
print('F1-score: %.3f (+/- %.3f )' % (scores.mean(), 2*scores.std()))



