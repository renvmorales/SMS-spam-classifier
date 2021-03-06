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


print('\nCreating a bar plot with most common words counts ....')
print('These are the %d words shown:' % len(common_words.index[common_words>min_value]))
print(list(common_words.index[common_words>min_value]))

# create barplot using seaborn
gr = sns.barplot(x=common_words.index[common_words>min_value], 
	y=common_words[common_words>min_value].values, palette='deep')

# rotate horizontal ticks (words)
gr.set_xticklabels(gr.get_xticklabels(), rotation=90)

plt.ylabel('Counts')
# show bar plot 
plt.show()


#########################################################################
# Word cloud of most frequent words


text = []

# group all words as many times in one list
for i in range(common_words.shape[0]):
	text += [common_words.index[i]]*common_words[i]

# create a long string with all common words
text = ' '.join(text)


# create a custom word cloud
wordcloud = WordCloud(max_words=50, width=800, height=400, 
	collocations=False, stopwords=[],
	background_color='white').generate(text)


print('\nCreating a word cloud of all common words ...')

# display word cloud
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#########################################################################
# count plot of normal and spam messages per month


# change indexes to datetimes values
df.index = pd.to_datetime(df['Date'],format='%Y-%m-%d %H:%M:%S')


# group by month and 'IsSpam' variable 
monthly = df.groupby([pd.TimeGrouper('M'), 'IsSpam']).count()


# bar plot using 
ax = monthly['Word_Count'].unstack(level=0).plot(kind='bar')
# sns.barplot(data=monthly, x=monthly.index)


# add a better description for the legend labels
ax.legend(['2017-Jan', '2017-Feb', '2017-Mar'])
plt.show()


#########################################################################
# Max, min, mean, median, std, variance of monthly 'Word_Count'


# group dataframe by month
monthly_word = df.groupby(pd.TimeGrouper('M'))


# aggregate 'Word_Count' values for general statistics
monthly_word = monthly_word['Word_Count'].agg(['min', 'max', 
	'mean', 'median', 'std'])


# change the indexes for better description
monthly_word.index = ['2017-Jan', '2017-Feb', '2017-Mar']


print('\nSome general statistics for monthly word counts:')
print(monthly_word)


#########################################################################
# days with largest non spam messages


# group dataframe by day
daily_word = df.groupby([pd.TimeGrouper('D'), 'IsSpam']).count()


daily_word = daily_word['Word_Count'].unstack(level=0)

non_spam_days = daily_word.loc['no']


# days with largest non spam messages
max_non_spam = non_spam_days.groupby(pd.TimeGrouper('M')).max()


print('\nDays with largest non spam messages (check manually):')
print(max_non_spam)