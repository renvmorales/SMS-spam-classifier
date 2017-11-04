#!/usr/bin/python3

import pandas as pd 
import numpy as np
import string



# read SMS.csv file 
df = pd.read_csv('SMS.csv', encoding = "ISO-8859-1")
print('\nReading csv file ....')



# group all messages in one list
messages = list(df['Full_Text'])


# a string with all punctuation characters
punc = string.punctuation


# remove all punctuation from all messages
nopunc_msg = []
for msg in messages:
	nopunc = [char for char in msg if char not in punc]
	nopunc_msg.append(''.join(nopunc))



# remove all common words (stopwords)


