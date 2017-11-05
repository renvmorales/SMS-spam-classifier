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


# shuffle data
X, Y = shuffle(df['Full_Text'], df['IsSpam'])



# function to convert into numerical labels
def binary_class(x):
	if x=='yes':
		return 1
	else:
		return 0

# convert all labels to 0 or 1
Y = Y.apply(binary_class)




# create a pipeline
pipeline = Pipeline([
	('bow', CountVectorizer(analyzer=text_process)),
	('tfidf', TfidfTransformer()),
	('classifier', MultinomialNB())
	])



# define the number for K-fold cross validation
kfold=10

print('\nApplying %d-fold cross-validation ...' % kfold)
print('\n')



# estimate f1-score using kfold cross validation 
f1_scores = cross_val_score(pipeline, X, Y, 
	cv=kfold, scoring='f1')

print('F1-score: %.3f (+/- %.3f)' % (f1_scores.mean(), 
	2*f1_scores.std()))



# estimate precision using kfold cross validation 
precisions = cross_val_score(pipeline, X, Y, 
	cv=kfold, scoring='precision')

print('Precision: %.3f (+/- %.3f)' % (precisions.mean(), 
	2*precisions.std()))



# estimate recall using kfold cross validation 
recalls = cross_val_score(pipeline, X, Y, 
	cv=kfold, scoring='recall')

print('Recall: %.3f (+/- %.3f)' % (recalls.mean(), 
	2*recalls.std()))
