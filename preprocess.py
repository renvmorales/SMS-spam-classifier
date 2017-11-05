#!/usr/bin/python3

import string
from nltk.corpus import stopwords



# define a function that can be used for text processing in a pipeline
def text_process(msg):

# a string with all punctuation characters	
	punc = string.punctuation

# remove all punctuation from the string message
	nopunc = [char for char in msg if char not in punc]

# join all characters into a single string
	nopunc = ''.join(nopunc)

# common words in english (to remove)
	remove = stopwords.words('english')

# return a list with all relevant words 
	return [word for word in nopunc.split() if word.lower() not in remove]



