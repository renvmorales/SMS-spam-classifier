# SMS-spam-classifier

## Introduction
Text mining is commonly used to provide some general description of unustructured data (e.g, single text or multiple 
collected user messages). This operation is important not just for exploratory data analysis purposes, but also when 
preprocesing data, a required step in order to feed a machine learning technique such as a classification algorithm. 
Here a small dataset of 5547 SMS messages (4827 regular and 747 spam) is briefly analyzed and converted numerically using tf-idf.
The goal here is to evaluate the generalization capacity of the following classification algorithms: Naïve Bayes, logistic regression, support vector machine (SVM), and multi-layer perceptron (MLP) neural networks. A 10-fold cross-validation approach is implemented using Python3 programming language, using the following modules: numpy, pandas, scikit-learn, wordcloud, nltk, string


## Directions

Run the exploratory data analysis:
```bash
	python3 EDA.py
```

So you can get some of the plots like this one:

<p align="center">
  <img src="word_barplot.png">
</p>




