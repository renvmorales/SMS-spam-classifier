#!/usr/bin/python3

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# read SMS.csv file 
df = pd.read_csv('SMS.csv', encoding = "ISO-8859-1")
print('\nReading csv file ....')

# print(df.head())


# messages = list(df['Full_Text'])
# print(messages[0:10])


print('\nTotal number of rows: %d' % df.shape[0])
print('Total number of columns: %d' % df.shape[1])


print('\nTotal normal messages: %d' % (df['IsSpam']=='no').sum())
print('Total spam messages: %d' % (df['IsSpam']=='yes').sum())




#########################################################################
# Bar plot of most frequent words


# compute total frequency of common words
common_words = (df.ix[:, 'got':'wan']).sum(axis=0)
# print(common_words.index)
# print(list(common_words.values))



# set style with ticks 
sns.set_style('ticks')

# minimum word count to display at the bar plot
min_value = 150

# create barplot using seaborn
gr = sns.barplot(x=common_words[common_words>min_value].index, 
	y=common_words[common_words>min_value].values, palette='deep')

# rotate horizontal ticks (words)
gr.set_xticklabels(gr.get_xticklabels(), rotation=90)

# show bar plot 
plt.show()


#########################################################################
# Word cloud 


