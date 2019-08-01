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
- Equations
	- precision = TP / (TP + FP)
	- recall = TP / (TP + FN) ==sensitivity
	- specificity = TN / (TN + FP)
	- accuracy = (TP + TN) / (TP + FP + TN + FN)
	- F1 Score = 2*P*R/(P+R)  --------P:precision R:recall
- 100% sensitive model: FN = 0, in other words, do not miss a TP but at a risk of having a lot of FP
- 100% specific model: FP = 0, do not miss a TN but at a risk of having a lot of FN
- 100% precise model: FP =0, do not miss a TN but at a risk of having a lot of FN

### Enforcement Learning 强化学习
- MDP: 系统状态 动作 奖励
- 深度学习：监督学习 非监督学习
- 区别
	- 强化学习得到的收益是短期的 训练过程并没有一个固定的监督机制
	- 与非监督学习相比存在一个小的短期反馈
	- 有独特的主动探索以及获得反馈的过程

### Semi Supervised Learning 半监督学习
- Defination: input包括一组label好的数据R和没有label好的数据U (一般情况下认为U的数量应该远大于R)
- types: transductive learning; inductive learning
	- Transuductive Learning: unlabeled data is the testing data
	- Inductive Learning: unlabeled data is not the test data
- Unsupervised Learning is usually used with some assmuptions. The effect of the final model depends on the correctness of your assumptions.

### Recurrent Neural Network (RNN)
- Definiation: 最重要的改进在于加入对weight的记忆功能
- 

### Long Short-term Memory (LSTM)
- four inputs; one output
- Structure
	- Input data
	- Input gate signal
	- Forget gate signal 
	- Output gate signal
- 跟RNN相比记忆时间更长


### Tutorials and Essays followed
- NLP的游戏规则从此改写？从word2vec, ELMo到BERT: https://zhuanlan.zhihu.com/p/47488095
- 如何评价 BERT 模型？: https://www.zhihu.com/question/298203515
- 词向量技术-从word2vec到Glove到ELMo https://blog.csdn.net/weixin_37947156/article/details/83146141
- NLP︱高级词向量表达（一）——GloVe（理论、相关测评结果、R&python实现、相关应用） https://blog.csdn.net/sinat_26917383/article/details/54847240
- Word Net中的名词 http://ccl.pku.edu.cn/doubtfire/semantics/wordnet/c-wordnet/nouns-in-wordnet.htm
- Clean your data with unsupervised machine learning https://towardsdatascience.com/clean-your-data-with-unsupervised-machine-learning-8491af733595
- 在sklearn.model_selection.GridSearchCV中使用自定义验证集进行模型调参: https://blog.csdn.net/isMarvellous/article/details/78195010
- Text classification using a few labeled examples https://www.researchgate.net/publication/256473085_Text_classification_using_a_few_labeled_examples
- Kmeans算法与KNN算法的区别 https://www.cnblogs.com/peizhe123/p/4619066.html
- 人工智能学习算法分类 https://www.cnblogs.com/peizhe123/p/9945438.html

- Unsupervised Learning
	- Automatic Text Categorization by Unsupervised Learning: https://aclweb.org/anthology/C00-1066
	- 基于句嵌入的无监督文本总结 https://www.jqr.com/article/000620
	- Unsupervised Text Summarization using Sentence Embeddings https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1


- Stuff to improve
	- Category generating based on comparing new data with previous one
	- Ground Truth Table
	- Predicting time

	- Confidential Level

### Plan for Next
plan for 2019.6.27
1. Test the performances for eda NLP with 6000 comments in total
2. Search and learn about the k-nearest analysis for words
3. Perform the k-nearest anaylysis on different categories
4. Think about how to build the ground truth table
s
### Essays and Leanging Poingts
#### That’s So Annoying!!!: A Lexical and Frame-Semantic Embedding Based Data Augmentation Approach to Automatic Categorization of Annoying Behaviors using #petpeeve Tweets 
Key points :
1. There are fixed methdos for doing KNN analysis for words
2. The augumentation is done by replacing KNN meaning-similar words in the original sentence.

#### Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks


#### Clean your data with unsupervised machine learning
- 核心思路是将文本中的单词和字符数量分别作为feature 以这两个feature作为依据进行 clustering
- 很有意思的思路 但是并不适用于当下问题  对于文本中描述的情况（读取HTML和PDF出现乱码 筛出乱码数据）比较有用

#### 李宏毅 Semi-supervised Learning
Outlines:
1. Semi-supervised Learning for Generative Model
2. Low-density Seperation Assumption
3. Smoothness Assumption
4. Better Representation (比较有用的一点)

- Supervised Generative Model
	 - 适用于Generative model
	 - 基本思路在于每一次训练完成后 根据训练出的模型对未标记数据进行预测 按照预测结果调整模型中的参数
	 - 类似于EM算法
	 - 从数学角度上来讲必定收敛 但收敛的结果受初始值影响

- Low-density Separation
"非黑即白"
- 假设不同数据之间界限非常明显
- 典型例子： Self Training
	- 可以用于大部分方法
	- 基本步骤：利用模型训练无标签数据 得到data set with presudo-label
	- Train model f from labelled data set
	- Apply f to the unlabeled data set
	- Remove a set of data from unlabeled data set, and add them into unlabeled data set (标准自己定 可以给一些data提供confidence)
	- Regression不能用这个
	- Soft label: output为几率 分别属于两个class的几率
	- Hard label：output必定为某一类的几率
	- 使用neural network一定要用hard

- 高级：Entropy-based regularization
	- neural netword结果为distribution 做出的假设是这个ditribution非常集中
	- 如何判断结果是不是集中: define the entropy of y - evaluate how concentrate the distribution y is
	![avatar](1/buildWebsites.jpg) 


- Build a general framework to combien the three layers together
- Search a few data augumentation method, manage to increase the model's ability in identifying minor class data
- Build a ground truth table for layer two robocall/telemarketer
- Searched about clustering method to do data cleaning and augumentation

### SMOTE:
合成少量类别过采样技术 基于随机过采样的一种改进方案 具体方法是在少量类内部选取一个样本X 计算其到其他同类样本中的欧式距离 介于二者之间取点作为新样本 优势在于解决了直接随机过采样（简单复制）的过拟合问题 缺陷在于算法中需要定义的参数K只有经过经验与实验才能得到 无法快速得到最优算法 另一方面取的样本中心x具有不确定性 如果刚好取到类别交界处 会进一步模糊不同类别间的交界导致算法准确率下降

### K-means 算法：
- 无监督学习聚类算法 具体做法在于先任意选取k个样本作为中心点 然后对于每一个数据选取与之距离最近的中心点作为自己的类别 第一次分类完成后再取每组数据的中心点作为新中心点重复上述操作 直到结果收敛
- 本质就是通过移动中心点逐渐"逼近"数据的中心
- 初始化的中心点对于最后的分类结果影响很大，因而很容易出现：当初始化的中心点不同时，其结果可能千差万别；
- 与KNN的区别和联系：练习是二者均可用于文本分类 但KNN更多用于supervised而K-means多用于聚类 KNN的思路是通过比较测试点与现有分组中的每个数据的距离 以最近的几个距离为依据进行分组 K-means的思路在于每次随机选取中点 然后以点距作为依据聚类 直到整体误差平方和最小

### 思路继续：
- 目前0.7的baseline说明现在的分类情况在大方向上有一定合理性 剩下的目标在于进一步优化数据分组
- 从修改第一层得到的经验当中看 downsampling比较有效 从第三层的经验来看 对于LSTN up sampling效果也不错
- 或者也可以考虑用之前用到的相似度对比方法移除！距离其他两类距离比较近的样本！

### Overview of Text Similarity Metrics in Python: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
techniques for sentence or document similarity

- Jaccard Similarity
	- Jaccard Similarity or Intersection over Union is defined as size of intersection divided by size of union of two sets
	![Jaccard Similarity](pictures/!Jaccard.png)
	- usually first perform lemmatization to reduce words to the same root word
	- In the example shown in the picture, Jaccard Similarity = 5/(5+3+2) = 0.5

- Cosine Similarity
	- Cosine Similarity calculates similarity by measuring the cosine of angle between two vectors. (向量成绩除以模之乘积)
	- First, we need to convert sentences into vectors. Vectorizing methods: beg of words, TF-IDF, Word2Vec

- Differences between Jaccard Similarity and Cosine Simialrity
	Jaccard similarity is good for cases where duplication does not matter, cosine similarity is good for cases where duplication matters while analyzing text similarity. For two product descriptions, it will be better to use Jaccard similarity as repetition of a word does not reduce their similarity.

### KNN clasiification
- 思路是先将文本转化为KNN中的synset（由一组同义词组成的list） 分别比较某句话中某个词跟另一句话中每一个词的语义相似程度 取最大分数为这个词的权重代表 取所有词的权重代表均值为句子的权重代表 (乍一看很靠谱......效果待验证)

- Sparse matrix: a matrix in which most of the elements are zero; by contrast, if most of the elements are nonzero, then the matrix is considered dense.

### 信息总结
- Symantec团队内部采用angle开发模式 同时注重研究和应用 一般不会由某人负责某一块工作很长时间 每两到三年会有一次工作方向转变
- 团队内部一般每两周进行一次review 确定下一步的工作方向
- 看重的品质：有做好事情的意愿和责任感，有做好事情的能力
- 工作能力大多数情况下都是学生时代的延续和积累 学生时代表现不佳工作之后获得巨大提升的情况很少
- Symantec的STAR部门主要负责的是核心engine的开发 Norton等其他部门往往直接
- 具体某方面编程的知识并不是毕业生的瓶颈 作为优秀的工程师很多知识都是可以后天掌握的 但是算法和编程思想等基本能力是非常关键的

### Kmeans practical implementation


### Text Clustering in Deep
Topic Modeling with LSA, PLSA, LDA & lda2Vec: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05

### Useful tutorial
手把手教你如何把jupyter notebook切换到其他配置好的conda虚拟环境: https://blog.csdn.net/weixin_41813895/article/details/84750990
好玩的Python库tqdm: https://blog.csdn.net/zejianli/article/details/77915751
Evolutionary Data Measures: Understanding the Difficulty of Text
Classification Tasks:https://www.aclweb.org/anthology/K18-1037

### Bert Usage
- Tokenisation: The tokenisation of bert involves splitting the input text into list of tokens that are available in the vocabulary in the pre-training set. To deal with the words not available in the vocabulary, BERT uses a technique called BPE. In this approcach an out of vocabulary word is progessively subwords. (In the author's opinion, this approach is quite important)
- initialistion Problem with Word Embedding (traditioanl deep learning methods): Pre-trained knowledge about words is in embeddings. Everything else if "from" zero (other words,syntex ...). Require a lot of training samples.
- Improtant Concepts:
	- Byte-Pair Encoding: transfer unrecognize words into sub words included in Embeeding
	- Transformer Structure: forwarding structure based on CNN; Implemded with attention, which flows up throgh different words and focus on the relations among all of them
	- Unsuervised Laguage Tasks: Train the whole network on corpus of text to do simple tasks like predict next word or predict missing word.
	- Fine Tuning: Take a model pretrained on huge corpus. Do additional training on unlabelled data. Much better than training from draft.
	- Simliar projects: ELMo, ULMFiT, OpenAI


