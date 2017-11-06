#!/usr/bin/python3


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocess import text_process

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import time



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
	('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
	('classifier', MLPClassifier(hidden_layer_sizes=(50,50,50), 
		activation='relu', max_iter=50, tol=1e-4, learning_rate_init=0.01,
		alpha=0.0001))
	])




# define the number for K-fold cross validation
kfold=10

print('\nApplying %d-fold cross-validation using MLP neural networks ...' % kfold)
print('\n')




start = time.time()
# estimate f1-score using kfold cross validation 
f1_scores = cross_val_score(pipeline, X, Y, 
	cv=kfold, scoring='f1')

print('F1-score: %.3f (+/- %.3f)    time: %.3f sec' % (f1_scores.mean(), 
	2*f1_scores.std(), (time.time()-start)))



start = time.time()
# estimate precision using kfold cross validation 
precisions = cross_val_score(pipeline, X, Y, 
	cv=kfold, scoring='precision')

print('Precision: %.3f (+/- %.3f)    time: %.3f sec' % (precisions.mean(), 
	2*precisions.std(), (time.time()-start)))



start = time.time()
# estimate recall using kfold cross validation 
recalls = cross_val_score(pipeline, X, Y, 
	cv=kfold, scoring='recall')

print('Recall: %.3f (+/- %.3f)    time: %.3f sec' % (recalls.mean(), 
	2*recalls.std(), (time.time()-start)))
