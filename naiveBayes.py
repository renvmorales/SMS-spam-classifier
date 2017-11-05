#!/usr/bin/python3


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocess import text_process

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle



# read SMS.csv file 
df = pd.read_csv('SMS.csv', encoding = "ISO-8859-1")
print('\nReading csv file ....')



X, Y = shuffle(df['Full_Text'], df['IsSpam'])





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


kfold=10

print('\nApplying %d-fold cross-validation ...' % kfold)
print('\n')


scores = cross_val_score(pipeline, X, Y.apply(binary_class), 
	cv=kfold, scoring='f1')

print('F1-score: %.3f (+/- %.3f)' % (scores.mean(), 2*scores.std()))



precisions = cross_val_score(pipeline, X, Y.apply(binary_class), 
	cv=kfold, scoring='precision')

print('Precision: %.3f (+/- %.3f)' % (precisions.mean(), 
	2*precisions.std()))



recalls = cross_val_score(pipeline, X, Y.apply(binary_class), 
	cv=kfold, scoring='recall')

print('Recall: %.3f (+/- %.3f)' % (recalls.mean(), 
	2*recalls.std()))
