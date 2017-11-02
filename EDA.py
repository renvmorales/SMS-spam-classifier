#!/usr/bin/python3

import pandas as pd 
import numpy as np
import seaborn as sns


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



