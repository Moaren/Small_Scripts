# TODO LIST for the phone number database project

## Problems to slove
- What to deal with?
- How to achive it?
- How is the performance?

### What to deal with?
How to define the labels, training sets and validation sets?
- TODO list:
	- Read through some comments from 800notes and try to analyze its pattern
	- Search more methods related with this context
	- Ask experienced people around

### Keywords
- topic modelling
- text classification

### Stuff to try
- https://www.aclweb.org/anthology/W14-2715
- https://www.jmir.org/2009/3/e25/
- https://ieeexplore.ieee.org/abstract/document/5276756
- https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=forum+classification+text+processing&btnG=
- https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
- ML common methods: http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/
- ML common method code: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
- Bert(Transfer Learning)
- Embedding
- Crossvalidatin(technique in data classifying)
- Softmax
- unsupervised classification: https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52
- SVM - said to have the best effect

### Pre-defined Categories
- Debt collector
- Event Reminder
- Fax
- Non-profit Organization
- Political Call
- Prank
- Scam Suspicion
- Silent Call
- Survey
- Telemarketers
- Text Message
- Other Unwanted
- Other Valid
Classification:
- negative
	- Debt collector(in practise)
	- Prank
	- Scam Suspicion
	- Silent Call
	- Telemarketers
	- Other Unwanted
- neutral or positive
	- Event Reminder
	- Fax
	- Non-profit Organization
	- Political Call
	- Survey
	- Text Message
	- Other Unwanted

### Things to do next (5.23)
- [ ] Fix the crawler and see if there is a better way to store the history queue and data (5.23)
- [ ] Look thourgh the LDA tutorial again and try that with current data (5.23 - 24)
- [ ] Look through Bert and Cross Validation for the supervised clasifying and apply it (by 5.29 Tuesday)
