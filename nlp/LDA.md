# Latent Dirichlet Allocation
## LDA Topic Models
- URL: https://www.youtube.com/watch?v=3mHy4OSyRf0&t=743s
- content summary: idea behind, why does it word, what tools and frameworks can be used here, tunable parameters, how to evaluate
- gensim: open source topic modeling frameword made in Python
- Preprocessing:
	- remove words occuring more than 10% and less than 10%
	- stop lists
	- lemmatization
	- parts of speech(noun,verb,adv) 
- Hypermeters: alpha, beta (parameters of Dirichlet distribution)
	- alpha controlling per document topic distribution, larger means the document tends to contain more topics and not just any single topic specifically
	- beta: larger beta, each topic is likely to contain a mixture of most of the words
	- high alpha --- make the documents to be more similar to each other
	- high beta --- make the topics to be more similar to be each other
- The author got around 50-200 clusters and labelled them in a space. Then he measured the Jesnsen-Sahnnon Divergence between them and clustered simialr groups together.
- The threshold can be done by doing a visual evaluation. (Compare the real stuff we try to classify and find where the threshold is.) 
- http://www.sohu.com/a/239937665_633698
- python下进行lda主题挖掘(二)——利用gensim训练LDA模型: https://blog.csdn.net/qq_23926575/article/details/79429689
- models.ldamulticore – parallelized Latent Dirichlet Allocation: https://radimrehurek.com/gensim/models/ldamulticore.html

