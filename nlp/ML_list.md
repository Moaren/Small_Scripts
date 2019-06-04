### Multi-Class Text Classification with Scikit-Learn
- The algorithm can only handle the number representation of the data, wich is measured throuh Term Frequency and Inverse Documenta Frequency (tf-idf) in this example.
How to calculate a tf-idf vector for each comment
```python
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
```

### Basic Stuff
- The false positive rate: 第一个形容词表示事实真假 第二个形容词表示人为偏好
- Standards in Machine Learning
	- precision: the ability to distinguish negative samples
	- recall: the ability to distinguish positive samples
	- F1-score: a combination of both

### Enforcement Learning 强化学习
- MDP: 系统状态 动作 奖励
- 深度学习：监督学习 非监督学习
- 区别
	- 强化学习得到的收益是短期的 训练过程并没有一个固定的监督机制
	- 与非监督学习相比存在一个小的短期反馈
	- 有独特的主动探索以及获得反馈的过程