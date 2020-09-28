# Databricks notebook source
# MAGIC %md # HW 3 - Synonym Detection In Spark
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC In the last homework assignment you performed Naive Bayes to classify documents as 'ham' or 'spam.' In doing so, we relied on the implicit assumption that the list of words in a document can tell us something about the nature of that document's content. We'll rely on a similar intuition this week: the idea that, if we analyze a large enough corpus of text, the list of words that appear in small window before or after a vocabulary term can tell us something about that term's meaning.
# MAGIC 
# MAGIC This will be your first assignment working in Spark. You'll perform Synonym Detection by repurposing an algorithm commonly used in Natural Language Processing to perform document similarity analysis. In doing so you'll also become familiar with important datatypes for efficiently processing sparse vectors and a number of set similarity metrics (e.g. Cosine, Jaccard, Dice). By the end of this homework you should be able to:  
# MAGIC * ... __define__ the terms `one-hot encoding`, `co-occurrance matrix`, `stripe`, `inverted index`, `postings`, and `basis vocabulary` in the context of both synonym detection and document similarity analysis.
# MAGIC * ... __explain__ the reasoning behind using a word stripe to compare word meanings.
# MAGIC * ... __identify__ what makes set-similarity calculations computationally challenging.
# MAGIC * ... __implement__ stateless algorithms in Spark to build stripes, inverted index and compute similarity metrics.
# MAGIC * ... __apply__ appropriate metrics to assess the performance of your synonym detection algorithm. 
# MAGIC 
# MAGIC 
# MAGIC __`NOTE`__: your reading assignment for weeks 5 and 6 were fairly heavy and you may have glossed over the papers on dimension independent similarity metrics by [Zadeh et al](http://stanford.edu/~rezab/papers/disco.pdf) and pairwise document similarity by [Elsayed et al](https://terpconnect.umd.edu/~oard/pdf/acl08elsayed2.pdf). If you haven't already, this would be a good time to review those readings -- they are directly relevant to this assignment.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__

# COMMAND ----------

# MAGIC %md # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.

# COMMAND ----------

import re
import ast
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ### Run the next cell to create your directory in dbfs
# MAGIC You do not need to understand this scala snippet. It simply dynamically fetches your user directory name so that any files you write can be saved in your own directory.

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
hw3_path = userhome + "/demo6/" 
hw3_path_open = '/dbfs' + hw3_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(hw3_path)

# COMMAND ----------

# RUN THIS CELL AS IS - A test to make sure your directory is working as expected.
# You should see a result like:
# dbfs:/user/youremail@ischool.berkeley.edu/hw3/sample_docs.txt
dbutils.fs.put(hw3_path+'test.txt',"hello world",True)
display(dbutils.fs.ls(hw3_path))

# COMMAND ----------

# RUN THIS CELL AS IS. You should see multiple google-eng-all-5gram-* files in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls('/mnt/mids-w261/data/HW3/'))

# COMMAND ----------

# get Spark Session info (RUN THIS CELL AS IS)
spark

# COMMAND ----------

# start SparkContext (RUN THIS CELL AS IS)
sc = spark.sparkContext

# COMMAND ----------

# OPTIONAL
# Spark configuration Information (RUN THIS CELL AS IS)
# sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md __`REMINDER:`__ If you are running this notebook in databricks, you can monitor the progress of your jobs using the Spark UI by clicking on "view" in the output cell below the cell you are running

# COMMAND ----------

# MAGIC %md # Question 1: Spark Basics.
# MAGIC In your readings and live session demos for weeks 4 and 5 you got a crash course in working with Spark. We also talked about how Spark RDDs fit into the broader picture of distributed algorithm design. The questions below cover key points from these discussions. Feel free to answer each one very briefly.
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ What is Spark? How  does it relate to Hadoop MapReduce?
# MAGIC 
# MAGIC * __b) short response:__ In what ways does Spark follow the principles of statelessness (a.k.a. functional programming)? List at least one way in which it allows the programmer to depart from this principle. 
# MAGIC 
# MAGIC * __c) short response:__ In the context of Spark what is a 'DAG' and how do they relate to the difference between an 'action' and a 'transformation'? Why is it useful to pay attention to the DAG that underlies your Spark implementation?
# MAGIC 
# MAGIC * __d) short response:__ Give a specific example of when we would want to `cache()` an RDD and explain why.

# COMMAND ----------

# MAGIC %md ### Q1 Student Answers:
# MAGIC > __a)__ Spark is a data processing framework that takes principles from MapReduce in that it employes distributed data and parallel computation and performs fast computation in *memory*. Features of Spark include low latency, optimized parallel communication patterns, improved shuffle, and efficient recovery from failures and straggler mitigation.
# MAGIC 
# MAGIC > __b)__ Spark follows principles of statelessness because it does not modify input data. Input data is read and new outputs are generated. The RDDs that make up the foundational data structure of Spark are readable and immutable, consistent with statelessness. All this being said, Spark does have the ability to deviate from functional programming. Caching data or intermediate results are examples of depart from functional programming.
# MAGIC 
# MAGIC > __c)__ A 'DAG' is a directed acyclic graph, which is essentially a schematic or set of physical instructions on how to build an RDD. It implies sequential (directed), non-looping (acyclic) nature of RDD construction, and can be used to trace the logical instructions of an RDD through its precursors (lineage). It relates to the difference between "transformations" and "actions" in that trasnformations are inherently structured as part of a DAG, as transformations are RDD-building operations and output RDDs. Actions call forth the DAG operations to materialize the RDD, and output summary or other non-RDD outputs. The DAG allows us to apply multiple transformations prior to the action step (as opposed to the generally single map & reduce steps of MapReduce). Paying attention to the DAG that underlies the Spark implementation allows us to understand the sequence of transformations and the logical processes attached in the construction of the RDD, enabling recreation and fault tolerance.
# MAGIC 
# MAGIC > __d)__ Caching refers to storing intermediate results in memory. In general, caching will be useful if the same dataset will be used multiple times in one program; for instance, in machine learning, multiple operations over the training data may be required, and caching may help preclude the need to process the same data from scratch at the beginning.

# COMMAND ----------

# MAGIC %md # Question 2: Similarity Metrics
# MAGIC As mentioned in the introduction to this assignment, an intuitive way to compare the meaning of two documents is to compare the list of words they contain. Given a vocabulary $V$ (feature set) we would represent each document as a vector of `1`-s and `0`-s based on whether or not it contains each word in $V$. These "one-hot encoded" vector representations allow us to use math to identify similar documents. However like many NLP tasks the high-dimensionality of the feature space is a challenge... especially when we start to scale up the size and number of documents we want to compare.
# MAGIC 
# MAGIC In this question we'll look at a toy example of document similarity analysis. Consider these 3 'documents': 
# MAGIC ```
# MAGIC docA	the flight of a bumblebee
# MAGIC docB	the length of a flight
# MAGIC docC	buzzing bumblebee flight
# MAGIC ```
# MAGIC These documents have a total of $7$ unique words: 
# MAGIC >`a, bumblebee, buzzing, flight, length, of, the`.     
# MAGIC 
# MAGIC Given this vocabulary, the documents' vector representations are (note that one-hot encoded entries follow the order of the vocab list above):
# MAGIC 
# MAGIC ```
# MAGIC docA	[1,1,0,1,0,1,1]
# MAGIC docB	[1,0,0,1,1,1,1]
# MAGIC docC	[0,1,1,1,0,0,0]
# MAGIC ```  
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ The cosine similarity between two vectors is $\frac{A\cdot B}{\|A\|\|B\|}$. Explain what the the numerator and denominator of this calculation would represent in terms of word counts in documents A and B. 
# MAGIC 
# MAGIC * __b) short response:__ Explain how the Jaccard, Overlap and Dice metrics are similar/different to the calculation for cosine similarity. When would these metrics lead to different similarity rankings for a set of documents?
# MAGIC 
# MAGIC * __c) short response:__ Calculate the cosine similarity for each pair of documents in our toy corpus. Please use markdown and $\LaTeX$ to show your calcuations.  
# MAGIC 
# MAGIC * __d) short response:__ According to your calculations in `part c` which pair of documents are most similar in meaning? Does this match your expecatation from reading the documents? If not, speculate about why we might have gotten this result.
# MAGIC 
# MAGIC * __e) short response:__ In NLP common words like '`the`', '`of`', and '`a`' increase our feature space without adding a lot of signal about _semantic meaning_. Repeat your analysis from `part c` but this time ignore these three words in your calculations [__`TIP:`__ _to 'remove' stopwords just ignore the vector entries in columns corresponding to the words you wish to disregard_]. How do your results change?

# COMMAND ----------

# MAGIC %md ### Q2 Student Answers:
# MAGIC > __a)__ The numerator is the multipication product of the two feature (word) vectors, which is the sum of the element-wise products of two vectors. In this case, it means the total number of words in common among both documents. The denominator is the product of the sqaure root of each vector sqaured in the feature space; this has the effect of normalizing the dot product. For this example, it represents the product of the square root of the word count of each vector.
# MAGIC 
# MAGIC > __b)__ The Jaccard, Overlap, and Dice metrics are similar to each other since they all calculated by the presence or absence of terms in documents. The Overlap takes the the words appeared in each document, the Jaccard takes the union, and Dice takes the full term count. The Cosine Similarity measures similarity by vector angle, and so when taking vectors in order for better information retrieval. The Jaccard, Overlap, and Dice metrics do not account for term frequency. Therefore, when two documents have a high frequency of a specific set of terms, but low coincidence across all other terms, might register as high similarity under Cosine Similiarity, but lower under Jaccard, Overlap, and Dice similarity. Additionally, the document size / length, subsets of documents, and other general factors or scenarios might lead to diffierent similarity rankings for a set of documents in the Jaccard, Overlap, and Dice metrics.
# MAGIC 
# MAGIC > __c)__ 
# MAGIC 
# MAGIC For numerator:
# MAGIC \\(A \cdot B = 1(1) + 1(0) + 0(0) + 1(1) + 0(1) + 1(1) + 1(1) = 4 \\)
# MAGIC 
# MAGIC \\(A \cdot C = 1(0) + 1(1) + 0(1) + 1(1) + 0(0) + 1(0) + 1(0)= 2 \\)
# MAGIC 
# MAGIC \\(B \cdot C = 1(0) + 0(1) + 0(1) + 1(1) + 1(0) + 1(0) + 1(0) = 1 \\)
# MAGIC 
# MAGIC For denominator:
# MAGIC 
# MAGIC \\(|A| = \sqrt{1^2 + 1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 1^2} = \sqrt{5} \\)
# MAGIC 
# MAGIC \\(|B| = \sqrt{1^2 + 0^2 + 0^2 + 1^2 + 1^2 + 1^2 + 1^2} = \sqrt{5} \\)
# MAGIC 
# MAGIC \\(|C| = \sqrt{0^2 + 1^2 + 1^2 + 1^2 + 0^2 + 0^2 + 0^2} = \sqrt{3} \\)
# MAGIC 
# MAGIC Cosine Similarity A-B: \\(\dfrac{A \cdot B}{|A||B|} = \dfrac{4}{\sqrt{5*5}} = 0.8 \\)
# MAGIC 
# MAGIC Cosine Similarity A-C: \\(\dfrac{A \cdot C}{|A||C|} = \dfrac{2}{\sqrt{5*3}} \approx 0.52 \\)
# MAGIC 
# MAGIC Cosine Similarity B-C: \\(\dfrac{B \cdot C}{|B||C|} = \dfrac{1}{\sqrt{5*3}} \approx 0.26 \\)
# MAGIC 
# MAGIC 
# MAGIC > __d)__ According to results from cosine similarity calculated in Part C, the document pair A and B contains the most similar documents. Based on the words of each respective document, this would NOT match expectations. The result is based on similarity of 'filler' words that are common across documents A and B (but not C), and add little context or meaning to the respective documents. These words include 'a', 'the', and 'of'.
# MAGIC 
# MAGIC > __e)__
# MAGIC 
# MAGIC New vectors:
# MAGIC 
# MAGIC ```
# MAGIC docA = [1,0,1,0]
# MAGIC docB = [0,0,1,1]
# MAGIC docC = [1,1,1,0]
# MAGIC ```
# MAGIC 
# MAGIC \\(|A| = \sqrt{1^2 + 0^2 + 1^2 + 0^2} = \sqrt{2} \\)
# MAGIC 
# MAGIC \\(|B| = \sqrt{0^2 + 0^2 + 1^2 + 1^2} = \sqrt{2} \\)
# MAGIC 
# MAGIC \\(|C| = \sqrt{1^2 + 1^2 + 1^2 + 0^2} = \sqrt{3} \\)
# MAGIC 
# MAGIC \\(A \cdot B = 1(0) + 0(0) + 1(1) + 0(1) = 1 \\)
# MAGIC 
# MAGIC \\(A \cdot C = 1(1) + 0(1) + 1(1) + 0(0) = 2 \\)
# MAGIC 
# MAGIC \\(B \cdot C = 0(1) + 0(1) + 1(1) + 1(0) = 1 \\)
# MAGIC 
# MAGIC Cosine Similarity A-B: \\(\dfrac{A \cdot B}{|A||B|} = \dfrac{1}{\sqrt{2*2}} = 0.5 \\)
# MAGIC 
# MAGIC Cosine Similarity A-C: \\(\dfrac{A \cdot C}{|A||C|} = \dfrac{2}{\sqrt{2*3}} \approx 0.82 \\)
# MAGIC 
# MAGIC Cosine Similarity B-C: \\(\dfrac{B \cdot C}{|B||C|} = \dfrac{1}{\sqrt{2*3}} \approx 0.41 \\)
# MAGIC 
# MAGIC The results change to be more in line with expectations. Document pair A and C contains the most similar documents.

# COMMAND ----------

# MAGIC %md # Question 3: Synonym Detection Strategy
# MAGIC 
# MAGIC In the Synonym Detection task we want to compare the meaning of words, not documents. For clarity, lets call the words whose meaning we want to compare `terms`. If only we had a 'meaning document' for each `term` then we could easily use the document similarity strategy from Question 2 to figure out which `terms` have similar meaning (i.e. are 'synonyms'). Of course in order for that to work we'd have to reasonably believe that the words in these 'meaning documents' really do reflect the meaning of the `term`. For a good analysis we'd also need these 'meaning documents' to be fairly long -- the one or two sentence dictionary definition of a term isn't going to provide enough signal to distinguish between thousands and thousands of `term` meanings.
# MAGIC 
# MAGIC This is where the idea of co-occurrance comes in. Just like DocSim makes the assumption that words in a document tell us about the document's meaning, we're going to assume that the set of words that 'co-occur' within a small window around our term can tell us some thing about the meaning of that `term`. Remember that we're going to make this 'co-words' list (a.k.a. 'stripe') by looking at a large body of text. This stripe is our 'meaning document' in that it reflects all the kinds of situations in which our `term` gets used in real language. So another way to phrase our assumption is: we think `terms` that get used to complete lots of the same phrases probably have related meanings. This may seem like an odd assumption but computational linguists have found that it works surprisingly well in practice. Let's look at a toy example to build your intuition for why and how.
# MAGIC 
# MAGIC Consider the opening line of Charles Dickens' _A Tale of Two Cities_:

# COMMAND ----------

corpus = """It was the best of times, it was the worst of times, 
it was the age of wisdom it was the age of foolishness"""

# COMMAND ----------

# MAGIC %md There are a total of 10 unique words in this short 'corpus':

# COMMAND ----------

words = list(set(re.findall(r'\w+', corpus.lower())))
print(words)

# COMMAND ----------

# MAGIC %md But of these 10 words, 4 are so common that they probably don't tell us very much about meaning.

# COMMAND ----------

stopwords = ["it", "the", "was", "of"]

# COMMAND ----------

# MAGIC %md So we'll ignore these 'stop words' and we're left with a 6 word vocabulary:

# COMMAND ----------

vocab = sorted([w for w in words if w not in stopwords])
print(vocab)

# COMMAND ----------

# MAGIC %md Your goal in the tasks below is to asses, which of these six words are most related to each other in meaning -- based solely on this short two line body of text.
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Given this six word vocabulary, how many 'pairs' of words do we want to compare? More generally for a n-word vocabulary how many pairwise comparisons are there to make? 
# MAGIC 
# MAGIC * __b) code:__ In the space provided below, create a 'stripe' for each `term` in the vocabulary. This stripe should be the list of all other vocabulary words that occur within a __5 word window__ (two words on either side) of the `term`'s position in the original text.
# MAGIC 
# MAGIC * __c) code + short response:__ Complete the provided code to turn your stripes into a 1-hot encoded co-occurrence matrix. For our 6 word vocabulary how many entries are in this matrix? How many entries are zeros? 
# MAGIC 
# MAGIC * __d) code:__ Complete the provided code to loop over all pairs and compute their cosine similarity. Please do not modify the existing code, just add your own in the spot marked.
# MAGIC 
# MAGIC * __e) short response:__ Which pairs of words have the highest 'similarity' scores? Are these words 'synonyms' in the traditional sense? In what sense are their meanings 'similar'? Explain how our results are contingent on the input text. What would change if we had a much larger corpus?

# COMMAND ----------

# MAGIC %md ### Q3 Student Answers:
# MAGIC > __a)__ For 6 words, compare each two out of six words, it will be \\(C_6^2 = \frac{6!}{2!4!}= 15 \\) unique paris. For n words, it will be \\(C_n^2 = \frac{n!}{2!(n-2)!}= \frac{n*(n-1)}{2}\\) unique paris.
# MAGIC 
# MAGIC > __c)__ There are 36 (6x6) entries in this matrix (symmetrical). However, it is a sparse matrix, most of the values are 0s.
# MAGIC 
# MAGIC > __e)__ The pairs 'best-worst' and 'foolishness-wisdom' have the highest 'similarity scores'. They are actually antonyms instead of 'synonyms' in traditional sense. They are 'synonyms' because of the calculation only depends on word frequency in a very small sample and our results also depend on the structure of the input text. In this case, these two pairs are 'similar' because its in the same location and same frequency to the word 'times' and 'age'. If we had a much larger corpus, we'd get more clear and normal usage of the terms, and the similarity score for these opposite-meaning words would presumably decrease based on more context. Our vector size would probably increase substantially, and the number of available words would increase correspondingly.

# COMMAND ----------

# for convenience, here are the corpus & vocab list again (RUN THIS CELL AS IS)
print("CORPUS:")
print(corpus)
print('VOCAB:')
print(vocab)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/MIDS_CV/gh-pages/best-of-times.png" />

# COMMAND ----------

# part b - USE THE TEXT ABOVE TO COMPLETE EACH STRIPE
stripes = {'age':['wisdom','foolishness'], # example
           'best':['times'], # YOU FILL IN THE REST
           'foolishness':['age'],
           'times': ['best','worst'],
           'wisdom':['age'],
           'worst':['times']}

# COMMAND ----------

# part c - initializing an empty co-occurrence matrix (RUN THIS CELL AS IS)
co_matrix = pd.DataFrame({term: [0]*len(vocab) for term in vocab}, index = vocab, dtype=int)

# COMMAND ----------

# part c - FILL IN THE MISSING LINE so that this cell 1-hot encodes the co-occurrence matrix
for term, nbrs in stripes.items():
    for nbr in nbrs:
        pass
        ############# YOUR CODE HERE #################
        co_matrix[term][nbr] += 1
        ############# (END) YOUR CODE #################
co_matrix

# COMMAND ----------

# part e - FILL IN THE MISSING LINES to compute the cosine similarity between each pair of terms
for term1, term2 in itertools.combinations(vocab, 2):
    # one hot-encoded vectors
    v1 = co_matrix[term1]
    v2 = co_matrix[term2]
    
    # cosine similarity
    ############# YOUR CODE HERE #################
    csim = sum(v1*v2)/((sum(v1^2)**0.5)*(sum(v2^2)**0.5))
    ############# (END) YOUR CODE #################    
    
    print(f"{term1}-{term2}: {csim}")

# COMMAND ----------

# MAGIC %md # Question 4: Pairs and Stripes at Scale
# MAGIC 
# MAGIC As you read in the paper by Zadeh et al, the advantage of metrics like Cosine, Dice, Overlap and Jaccard is that they are dimension independent -- that is to say, if we implement them in a smart way the computational complexity of performing these computations is independent of the number of documents we want to compare (or in our case, the number of terms that are potential synonyms). One component of a 'smart implementation' involves thinking carefully both about how you define the "basis vocabulary" that forms your feature set (removing stopwords, etc). Another key idea is to use a data structure that facilitates distributed calculations. The DISCO implemetation further uses a sampling strategy, but that is beyond the scope of this assignment. 
# MAGIC 
# MAGIC In this question we'll take a closer look at the computational complexity of the synonym detection approach we took in question 3 and then revist the document similarity example as a way to explore a more efficient approach to parallelizing this analysis.
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ In question 3 you calculated the cosine similarity of pairs of words using the vector representation of their co-occurrences in a corpus. Imagine for now that you have unlimited memory on each of your nodes and describe a sequence of map & reduce steps that would start from a raw corpus and reproduce your strategy from Q3. For each step be sure to note what information would be stored in memory on your nodes and what information would need to be shuffled over the network (a bulleted list of steps with 1-2 sentences each is sufficient to answer this question).
# MAGIC 
# MAGIC * __b) short response:__ In the asynch videos about "Pairs and Stripes" you were introduced to an alternative strategy. Explain two ways that using these data structures are more efficient than 1-hot encoded vectors when it comes to distributed similarity calculations [__`HINT:`__ _Consider memory constraints, amount of information being shuffled, amount of information being transfered over the network, and level of parallelization._]
# MAGIC 
# MAGIC * __c) read provided code:__ The code below provides a streamined implementation of Document similarity analysis in Spark. Read through this code carefully. Once you are confident you understand how it works, answer the remaining questions. [__`TIP:`__ _to see the output of each transformation try commenting out the subsequent lines and adding an early `collect()` action_.]
# MAGIC 
# MAGIC * __d) short response:__ The second mapper function, `splitWords`, emits 'postings'. The list of all 'postings' for a word is also refered to as an 'inverted index'. In your own words, define each of these terms ('postings' and 'inverted index') based on your reading of the provided code. (*DITP by Lin and Dyer also contains a chapter on the Inverted Index although in the context of Hadoop rather than Spark*).
# MAGIC 
# MAGIC * __e) short response:__ The third mapper, `makeCompositeKeys`, loops over the inverted index to emit 'pairs' of what? Explain what information is included in the composite key created at this stage and why it makes sense to synchronize around that information in the context of performing document similarity calculations. In addition to the information included in these new keys, what other piece of information will we need to compute Jaccard or Cosine similarity?
# MAGIC 
# MAGIC * __f) short response:__ Out of all the Spark transformations we make in this analysis, which are 'wide' transformations and which are 'narrow' transformations. Explain.

# COMMAND ----------

# MAGIC %md ### Q4 Student Answers:
# MAGIC > __a)__
# MAGIC 
# MAGIC - Create RDD via sc.textfile and parallelize via sc.parallelize (i.e. enable parallel / partitioning)
# MAGIC - Map function: Single map function applied via flatmap to:
# MAGIC   1. Tokenize all the words. *(Stored in memory at nodes)*
# MAGIC   2. Filter the corpus by removing stopwords. *(Stored in memory at nodes)*
# MAGIC   3. Loop over all words, get all the combinations of pairs and emit tuple of ((word1,word2),1)) *(Emitted & shuffled over network)*
# MAGIC - Map Function: Reduce by Key: Add all counts for each word-pair key, emit new key-value pair (word1: (word2,count))
# MAGIC - Reducer Function: Reduce into one-hot encoded array using new key and calculate consime similarities
# MAGIC 
# MAGIC > __b)__ The alternative strategy is to group together pairs into associative arrays, which essentially utilize dictionaries to store associated words and their respective counts. These associative arrays greatly reduce the amount of shuffling, sorting, and information transferred over the network.  
# MAGIC 
# MAGIC > __c)__ _read provided code before answering d-f_ 
# MAGIC 
# MAGIC > __d)__ An inverted index refers to the swapping key-value pairs. In the general case we're discussing, it refers to a word and its mapped 'location', which is a document / label. In this particular case, when the documents are read in, the original key is the document label, and the value is the text. Here, we have inverted the index with the splitWords function to obtain a word as an index, with the document along with the number of words in the document as the value. Together this comprises the 'posting' for a word: the document in which the word appeared, along with the size of that document. A word can have multiple postings with the same document location or different document locations. The reduceByKey compiles these postings by word, and emits each word with all associated postings.
# MAGIC 
# MAGIC > __e)__ The makeCompositeKeys mapper emits pairs of documents (and their respective document sizes) that represent all non-repeating pair-wise combinations of the postings associated with each word. The new composite key therefore is comprised of the paired document names / ids. When performing document similarity calculations (via Jaccard, Dice, and / or Overlap), we are interested in seeing the presence (i.e. intersections and unions) of words in two documents being compared. Therefore it makes sense to synchronize information over document pairs and their associated unique word counts. We can imagine that the next phase will include a reduceByKey to compute this sum. The final piece of information needed to perform the Jaccard is the union of the two documents, which is the sum of the total number of words in each document minus the overlap.
# MAGIC 
# MAGIC > __f)__ Each of the .map and .flatmap transformations are 'narrow' transformations. They can be completed within a single partition, completely in parallel. The 'wide' transformations include the .reduceByKey transformations. The elements required to perform these transformations using many partitions. They require us to map across multiple partitions in order to achieve the key-based reduction.

# COMMAND ----------

# MAGIC %md A small test file: __`sample_docs.txt`__

# COMMAND ----------

# RUN THIS CELL AS IS
dbutils.fs.put(hw3_path+"sample_docs.txt", 
"""docA	bright blue butterfly forget
docB	best forget bright sky
docC	blue sky bright sun
docD	under butterfly sky hangs
docE	forget blue butterfly""", True)

# COMMAND ----------

# RUN THIS CELL AS IS
print(dbutils.fs.head(hw3_path+"sample_docs.txt"))

# COMMAND ----------

# MAGIC %md __Document Similarity Analysis in Spark:__

# COMMAND ----------

# load data - RUN THIS CELL AS IS
data = sc.textFile(hw3_path+"sample_docs.txt")  

# COMMAND ----------

# helper function - RUN THIS CELL AS IS
def splitWords(pair):
    """Mapper 2: tokenize each document and emit postings."""
    doc, text = pair
    words = text.split(" ")
    for w in words:
        yield (w, [(doc,len(words))])

# COMMAND ----------

# helper function - RUN THIS CELL AS IS
def makeCompositeKey(inverted_index):
    """Mapper 3: loop over postings and yield pairs."""
    word, postings = inverted_index
    # taking advantage of symmetry, output only (a,b), but not (b,a)
    for subset in itertools.combinations(sorted(postings), 2):
        yield (str(subset), 1)

# COMMAND ----------

# helper function - RUN THIS CELL AS IS
def jaccard(line):
    """Mapper 4: compute similarity scores"""
    (doc1, n1), (doc2, n2) = ast.literal_eval(line[0])
    total = int(line[1])
    jaccard = total / float(int(n1) + int(n2) - total)
    yield doc1+" - "+doc2, jaccard

# COMMAND ----------

# Spark Job - RUN THIS CELL AS IS
result = data.map(lambda line: line.split('\t')) \
             .flatMap(splitWords) \
             .reduceByKey(lambda x,y : x+y) \
             .flatMap(makeCompositeKey) \
             .reduceByKey(lambda x,y : x+y) \
             .flatMap(jaccard) \
             .takeOrdered(10, key=lambda x: -x[1])
result

# COMMAND ----------

# MAGIC %md # About the Data
# MAGIC Now that you are comfortable with similarity metrics we turn to the main task in this assignment: Synonym Detection. As you saw in Question 3 the ability of our algorithm to detect words with similar meanings is highly dependent on our input text. Specifically, we need a large enough corpus of natural language that we can expose our algorithm to a realistic range of contexts in which any given word might get used. Ideally, these 'contexts' would also provide enough signal to distinguish between words with similar semantic roles but different meaning. Finding such a corpus will be easier to accomplish for some words than others.
# MAGIC 
# MAGIC For the main task in this portion of the homework you will use data from Google's n-gram corpus. This data is particularly convenient for our task because Google has already done the first step for us: they windowed over a large subset of the web and extracted all 5-grams. If you are interested in learning more about this dataset the original source is: http://books.google.com/ngrams/, and a large subset is available [here from AWS](https://aws.amazon.com/datasets/google-books-ngrams/). 
# MAGIC 
# MAGIC For this assignment we have provided a subset of the 5-grams data consisting of 191 files of approximately 10MB each. These files are available in dbfs. Please only use the provided data so that we can ensure consistent results from student to student.
# MAGIC 
# MAGIC Each row in our dataset represents one of these 5 grams in the format:
# MAGIC > `(ngram) \t (count) \t (pages_count) \t (books_count)`
# MAGIC 
# MAGIC __DISCLAIMER__: In real life, we would calculate the stripes cooccurrence data from the raw text by windowing over the raw text and not from the 5-gram preprocessed data.  Calculating pairs on this 5-gram is a little corrupt as we will be double counting cooccurences. Having said that this exercise can still pull out some similar terms.

# COMMAND ----------

# RUN THIS CELL AS IS. You should see multiple google-eng-all-5gram-* files in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls('/mnt/mids-w261/data/HW3/'))

# COMMAND ----------

# set global paths to full data folder and to the first file (which we'll use for testing)
NGRAMS = '/mnt/mids-w261/data/HW3'
F1_PATH = '/mnt/mids-w261/data/HW3/googlebooks-eng-all-5gram-20090715-0-filtered.txt'

# COMMAND ----------

# MAGIC %md As you develop your code you should use the following file to systems test each of your solutions before running it on the Google data. (Note: these are the 5-grams extracted from our two line Dickens corpus in Question 3... you should find that your Spark job results match the calculations we did "by hand").
# MAGIC 
# MAGIC Test file: __`systems_test.txt`__

# COMMAND ----------

# RUN THIS CELL AS IS
dbutils.fs.put(hw3_path+"systems_test.txt",
"""it was the best of	1	1	1
age of wisdom it was	1	1	1
best of times it was	1	1	1
it was the age of	2	1	1
it was the worst of	1	1	1
of times it was the	2	1	1
of wisdom it was the	1	1	1
the age of wisdom it	1	1	1
the best of times it	1	1	1
the worst of times it	1	1	1
times it was the age	1	1	1
times it was the worst	1	1	1
was the age of wisdom	1	1	1
was the best of times	1	1	1
was the age of foolishness	1	1	1
was the worst of times	1	1	1
wisdom it was the age	1	1	1
worst of times it was	1	1	1""",True)

# COMMAND ----------

# MAGIC %md Finally, we'll create a Spark RDD for each of these files so that they're easy to access throughout the rest of the assignment.

# COMMAND ----------

# RUN THIS CELL AS IS Spark RDDs for each dataset
testRDD = sc.textFile(hw3_path+"systems_test.txt") 
f1RDD = sc.textFile(F1_PATH)
dataRDD = sc.textFile(NGRAMS)

# COMMAND ----------

# MAGIC %md Let's take a peak at what each of these RDDs looks like:

# COMMAND ----------

testRDD.take(10)

# COMMAND ----------

f1RDD.take(10)

# COMMAND ----------

dataRDD.take(10)

# COMMAND ----------

# MAGIC %md # Question 5: N-gram EDA part 1 (words)
# MAGIC 
# MAGIC Before starting our synonym-detection, let's get a sense for this data. As you saw in questions 3 and 4 the size of the vocabulary will impact the amount of computation we have to do. Write a Spark job that will accomplish the three tasks below as efficiently as possible. (No credit will be awarded for jobs that sort or subset after calling `collect()`-- use the framework to get the minimum information requested). As you develop your code, systems test each job on the provided file with Dickens ngrams, then on a single file from the Ngram dataset before running the full analysis.
# MAGIC 
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) code:__ Write a Spark application to retrieve:
# MAGIC   * The number of unique words that appear in the data. (i.e. size of the vocabulary) 
# MAGIC   * A list of the top 10 words & their counts.
# MAGIC   * A list of the bottom 10 words & their counts.  
# MAGIC   
# MAGIC   __`NOTE  1:`__ _don't forget to lower case the ngrams before extracting words._  
# MAGIC   __`NOTE  2:`__ _don't forget to take in to account the number of occurances of each ngram._  
# MAGIC   __`NOTE  3:`__ _to make this code more reusable, the `EDA1` function code base uses a parameter 'n' to specify the number of top/bottom words to print (in this case we've requested 10)._
# MAGIC 
# MAGIC 
# MAGIC * __b) short response:__ Given the vocab size you found in part a, how many potential synonym pairs could we form from this corpus? If each term's stripe were 1000 words long, how many individual 'postings' tuples would we need to shuffle inorder to form the inverted indices? Show and briefly explain your calculations for each part of this question. [__`HINT:`__ see your work from q4 for a review of these concepts.]
# MAGIC 
# MAGIC * __c) short response:__ What do you notice about the most frequent words, how usefull will these top words be in synonym detection? Explain.
# MAGIC 
# MAGIC * __d) short response:__ What do you notice/infer about the least frequent words, how reliable should we expect the detected 'synonyms' for the bottom words to be? Explain.

# COMMAND ----------

# MAGIC %md ### Q5 Student Answers:
# MAGIC 
# MAGIC > __b)__ The vocabulary size is 269,339. As calculated previously, there are 269339 x 269338/2 = 36271613791 potential synonym pairs. With 1000-word stripes, we would have to construct 1000 x 269339 = 269339000 individual postings tuples that would need to be shuffed to form the inverted indices. We can see this from the postings function
# MAGIC 
# MAGIC > __c)__ The most frequent words are the stop words. They are not very useful in synonym detection, as those stop words are generally meaningless except for grammer. Therefore, a large number of different words may show as similar due to proximity to these stop words, but will only be similar in their grammatical function, rather than their meaning. Still, with a large enough corpus, this could potentially be useful in contributing to similarity.
# MAGIC 
# MAGIC > __d)__ There are lots of proper nouns / names in the least frequent words. We would not expect the synonyms for bottom words to be reliable at all, as there are generally no synonyms for proper nouns. It may be that it can detect people with the same names that are equally famous for specific things. But, more than likely, it will be inaccurate.

# COMMAND ----------

# part a - write your spark job here 
def EDA1(rdd, n):
    total, top_n, bottom_n = None, None, None
    ############# YOUR CODE HERE ###############
    def counter(line):
      split = line.lower().split('\t')
      line = (split[0].split(' '),int(split[1]))
      for i in line[0]:
        yield (i,line[1])

    mapped_rdd = rdd.flatMap(counter) \
                    .reduceByKey(lambda a,b: a+b)

    total = mapped_rdd.count()
    top_n = mapped_rdd.takeOrdered(n, key=lambda x: -x[1])
    bottom_n = mapped_rdd.takeOrdered(n, key=lambda x: x[1])

    ############# (END) YOUR CODE ##############
    return total, top_n, bottom_n

# COMMAND ----------

# part a - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
import time
start = time.time()
vocab_size, most_frequent, least_frequent = EDA1(testRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 1.0887939929962158 seconds

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print("Vocabulary Size:", vocab_size)
print(" ---- Top Words ----|--- Bottom Words ----")
for (w1, c1), (w2, c2) in zip(most_frequent, least_frequent):
    print(f"{w1:>8} {c1:>10} |{w2:>15} {c2:>3}")

# COMMAND ----------

# MAGIC %md Expected output for testRDD:
# MAGIC <pre>
# MAGIC     Vocabulary Size: 10
# MAGIC  ---- Top Words ----|--- Bottom Words ----
# MAGIC      was         17 |    foolishness   1
# MAGIC       of         17 |           best   4
# MAGIC      the         17 |          worst   5
# MAGIC       it         16 |         wisdom   5
# MAGIC    times         10 |            age   8
# MAGIC      age          8 |          times  10
# MAGIC    worst          5 |             it  16
# MAGIC   wisdom          5 |            was  17
# MAGIC     best          4 |             of  17
# MAGIC foolishness       1 |            the  17  
# MAGIC </pre>

# COMMAND ----------

# part a - run a single file, ie., a small sample (RUN THIS CELL AS IS)
start = time.time()
vocab_size, most_frequent, least_frequent = EDA1(f1RDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 1.9411416053771973 seconds

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print("Vocabulary Size:", vocab_size)
print(" ---- Top Words ----|--- Bottom Words ----")
for (w1, c1), (w2, c2) in zip(most_frequent, least_frequent):
    print(f"{w1:>8} {c1:>10} |{w2:>15} {c2:>3}")

# COMMAND ----------

# MAGIC %md Expected output for f1RDD
# MAGIC <pre>
# MAGIC Vocabulary Size: 36353
# MAGIC  ---- Top Words ----|--- Bottom Words ----
# MAGIC      the   27691943 |    stakeholder  40
# MAGIC       of   18590950 |          kenny  40
# MAGIC       to   11601757 |         barnes  40
# MAGIC       in    7470912 |         arnall  40
# MAGIC        a    6926743 |     buonaparte  40
# MAGIC      and    6150529 |       puzzling  40
# MAGIC     that    4077421 |             hd  40
# MAGIC       is    4074864 |        corisca  40
# MAGIC       be    3720812 |       cristina  40
# MAGIC      was    2492074 |         durban  40
# MAGIC </pre>

# COMMAND ----------

# part a - run full analysis (RUN THIS CELL AS IS)
start = time.time()
vocab_size, most_frequent, least_frequent = EDA1(dataRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 148.80424451828003 seconds

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print("Vocabulary Size:", vocab_size)
print(" ---- Top Words ----|--- Bottom Words ----")
for (w1, c1), (w2, c2) in zip(most_frequent, least_frequent):
    print(f"{w1:>8} {c1:>10} |{w2:>15} {c2:>3}")

# COMMAND ----------

# MAGIC %md Expected output for dataRDD:
# MAGIC (bottom words might vary a little due to ties)
# MAGIC <pre>
# MAGIC Vocabulary Size: 269339
# MAGIC  ---- Top Words ----|--- Bottom Words ----
# MAGIC      the 5490815394 |   schwetzingen  40
# MAGIC       of 3698583299 |           cras  40
# MAGIC       to 2227866570 |       parcival  40
# MAGIC       in 1421312776 |          porti  40
# MAGIC        a 1361123022 |    scribbler's  40
# MAGIC      and 1149577477 |      washermen  40
# MAGIC     that  802921147 |    viscerating  40
# MAGIC       is  758328796 |         mildes  40
# MAGIC       be  688707130 |      scholared  40
# MAGIC       as  492170314 |       jaworski  40
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md # Question 6: N-gram EDA part 2 (co-occurrences)
# MAGIC 
# MAGIC The computational complexity of synonym analysis depends not only on the number of words, but also on the number of co-ocurrences each word has. In this question you'll take a closer look at that aspect of our data. As before, please test each job on small "systems test" (Dickens ngrams) file and on a single file from the Ngram dataset before running the full analysis.
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ Write a spark job that computes:
# MAGIC   * the number of unique neighbors (i.e. 5-gram co-occuring words) for each word in the vocabulary. 
# MAGIC   * the top 10 words with the most "neighbors"
# MAGIC   * the bottom 10 words with least "neighbors"
# MAGIC   * a random sample of 1% of the words' neighbor counts  
# MAGIC   __`NOTE:`__ for the last item, please return only the counts and not the words -- we'll go on to use these in a plotting function that expects a list of integers.
# MAGIC 
# MAGIC 
# MAGIC * __b) short response:__ Use the provided code to plot a histogram of the sampled list from `a`. Comment on the distribution you observe. How will this distribution affect our synonym detection analysis?
# MAGIC 
# MAGIC * __c) code + short response:__ Write a Spark Job to compare the top/bottom words from Q5 and from part a. Specifically, what % of the 1000 most/least neighbors words also appear in the list of 1000 most/least frequent words. [__`NOTE:`__ _technically these lists are short enough to comparing in memory on your local machine but please design your Spark job as if we were potentially comparing much larger lists._]

# COMMAND ----------

# MAGIC %md ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ The distribution is skewed with the most of the words have very few co-words. This is likely going to make it difficult to detect synonyms for the large majority of words in the corpus. As we saw in previous questions, a good number may include things like proper nouns, etc, which we would not expect to have synonyms. 
# MAGIC 
# MAGIC > __c)__ 88% of the words with the most neighbors appear in the list of 1000 most frequent words. 1.9% of the words with the least neighbors appear in the list of 1000 least frequent words.

# COMMAND ----------

# part a - spark job
def EDA2(rdd,n):
    top_n, bottom_n, sampled_counts = None, None, None
    ############# YOUR CODE HERE ###############
    def emitStripes(line):
      from collections import defaultdict
      line = line.split('\t')[0].lower().split(' ')
      for i in range(5):
        h = defaultdict(int)
        for j in range(5):
          if line[i] != line[j]:
            h[line[j]] += 0
          else:
            pass
        if len(h) > 0:
          yield (line[i],h)
        else:
          pass

    def sumStripes(stripe1, stripe2):
      for i in stripe2:
        stripe1[i] += 0
      return stripe1

    mapped_rdd = rdd.flatMap(emitStripes) \
                    .reduceByKey(sumStripes) \
                   .map(lambda x: (x[0], len(x[1])))

    top_n = mapped_rdd.takeOrdered(n, key=lambda x: -x[1])
    bottom_n = mapped_rdd.takeOrdered(n, key=lambda x: x[1])
    sampled_counts = mapped_rdd.map(lambda x: x[1]).sample(True,0.01).collect()
    
    ############# (END) YOUR CODE ##############
    return top_n, bottom_n, sampled_counts

# COMMAND ----------

# part a - systems test (RUN THIS CELL AS IS)
start = time.time()
most_nbrs, least_nbrs, sample_counts = EDA2(testRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 2.1029624938964844 seconds

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print(" --- Most Co-Words ---|--- Least Co-Words ----")
for (w1, c1), (w2, c2) in zip(most_nbrs, least_nbrs):
    print(f"{w1:>12} {c1:>8} |{w2:>16} {c2:>4}")

# COMMAND ----------

# MAGIC %md Expected output for testRDD:
# MAGIC <pre>
# MAGIC  --- Most Co-Words ---|--- Least Co-Words ----
# MAGIC          was        9 |     foolishness    4
# MAGIC           of        9 |            best    5
# MAGIC          the        9 |           worst    5
# MAGIC           it        8 |          wisdom    5
# MAGIC          age        7 |             age    7
# MAGIC        times        7 |           times    7
# MAGIC         best        5 |              it    8
# MAGIC        worst        5 |             was    9
# MAGIC       wisdom        5 |              of    9
# MAGIC  foolishness        4 |             the    9
# MAGIC  </pre>

# COMMAND ----------

# part a - single file test (RUN THIS CELL AS IS)
start = time.time()
most_nbrs, least_nbrs, sample_counts = EDA2(f1RDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 7.468951463699341 seconds

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print(" --- Most Co-Words ---|--- Least Co-Words ----")
for (w1, c1), (w2, c2) in zip(most_nbrs, least_nbrs):
    print(f"{w1:>12} {c1:>8} |{w2:>16} {c2:>4}")

# COMMAND ----------

# MAGIC %md Expected output for f1RDD:
# MAGIC <pre>
# MAGIC  --- Most Co-Words ---|--- Least Co-Words ----
# MAGIC          the    25548 |              vo    1
# MAGIC           of    22496 |      noncleaved    2
# MAGIC          and    16489 |        premiers    2
# MAGIC           to    14249 |        enclaves    2
# MAGIC           in    13891 |   selectiveness    2
# MAGIC            a    13045 |           trill    2
# MAGIC         that     8011 |           pizza    2
# MAGIC           is     7947 |            hoot    2
# MAGIC         with     7552 |     palpitation    2
# MAGIC           by     7400 |            twel    2
# MAGIC </pre>

# COMMAND ----------

# part a - full data (RUN THIS CELL AS IS)
start = time.time()
most_nbrs, least_nbrs, sample_counts = EDA2(dataRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 140.08682656288147 seconds

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print(" --- Most Co-Words ---|--- Least Co-Words ----")
for (w1, c1), (w2, c2) in zip(most_nbrs, least_nbrs):
    print(f"{w1:>12} {c1:>8} |{w2:>16} {c2:>4}")

# COMMAND ----------

# MAGIC %md Expected output for dataRDD: 
# MAGIC (bottom words might vary a little due to ties)
# MAGIC <pre>
# MAGIC  --- Most Co-Words ---|--- Least Co-Words ----
# MAGIC          the   164982 |          cococo    1
# MAGIC           of   155708 |            inin    1
# MAGIC          and   132814 |        charuhas    1
# MAGIC           in   110615 |         ooooooo    1
# MAGIC           to    94358 |           iiiii    1
# MAGIC            a    89197 |          iiiiii    1
# MAGIC           by    67266 |             cnj    1
# MAGIC         with    65127 |            choh    1
# MAGIC         that    61174 |             neg    1
# MAGIC           as    60652 |      cococococo    1
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md __`NOTE:`__ _before running the plotting code below, make sure that the variable_ `sample_counts` _points to the list generated in_ `part a`.

# COMMAND ----------

# part b - plot histogram (RUN THIS CELL AS IS - feel free to modify format)

# removing extreme upper tail for a better visual
counts = np.array(sample_counts)[np.array(sample_counts) < 6000]
t = sum(np.array(sample_counts) > 6000)
n = len(counts)
print("NOTE: we'll exclude the %s words with more than 6000 nbrs in this %s count sample." % (t,n))

# set up figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))

# plot regular hist
ax1.hist(counts, bins=50)
ax1.set_title('Freqency of Number of Co-Words', color='0.1')
ax1.set_facecolor('0.9')
ax1.tick_params(axis='both', colors='0.1')
ax1.grid(True)

# plot log scale hist
ax2.hist(counts, bins=50)
ax2.set_title('(log)Freqency of Number of Co-Words', color='0.1')
ax2.set_facecolor('0.9')
ax2.tick_params(axis='both', colors='0.1')
ax2.grid(True)
plt.yscale('log')


# COMMAND ----------

display(fig)

# COMMAND ----------

# part c - spark job
def compareRankings(rdd1, rdd2):
    percent_overlap = None
    ############# YOUR CODE HERE ###############
    intersect = rdd1.cogroup(rdd2) \
                    .filter(lambda x: x[1][0] and x[1][1]) \
                    .count()

    percent_overlap = 100*intersect/1000
    
    ############# (END) YOUR CODE ##############
    return percent_overlap

# COMMAND ----------

# part c - get lists for comparison (RUN THIS CELL AS IS...)
# (... then change 'testRDD' to 'f1RDD'/'dataRDD' when ready)
total, topWords, bottomWords = EDA1(dataRDD, 1000)
topNbrs, bottomNbrs, sample_counts = EDA2(dataRDD, 1000)
twRDD = sc.parallelize(topWords)
bwRDD = sc.parallelize(bottomWords)
tnRDD = sc.parallelize(topNbrs)
bnRDD = sc.parallelize(bottomNbrs)
top_overlap = compareRankings(tnRDD, twRDD)
bottom_overlap = compareRankings(bnRDD,bwRDD)
print(f"Of the 1000 words with most neighbors, {top_overlap} percent are also in the list of 1000 most frequent words.")
print(f"Of the 1000 words with least neighbors, {bottom_overlap} percent are also in the list of 1000 least frequent words.")

# COMMAND ----------

# MAGIC %md # Question 7: Basis Vocabulary & Stripes
# MAGIC 
# MAGIC Every word that appears in our data is a potential feature for our synonym detection analysis. However as we've discussed, some are likely to be more useful than others. In this question, you'll choose a judicious subset of these words to form our 'basis vocabulary' (i.e. feature set). Practically speaking, this means that when we build our stripes, we are only going to keep track of when a term co-occurs with one of these basis words. 
# MAGIC 
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) short response:__ Suppose we were deciding between two different basis vocabularies: the 1000 most frequent words or the 1000 least frequent words. How would this choice impact the quality of the synonyms we are able to detect? How does this choice relate to the ideas of 'overfitting' or 'underfitting' a training set?
# MAGIC 
# MAGIC * __b) short response:__ If we had a much larger dataset, computing the full ordered list of words would be extremely expensive. If we need to none-the-less get an estimate of word frequency in order to decide on a basis vocabulary (feature set), what alternative strategy could we take?
# MAGIC 
# MAGIC * __c) code:__ Write a spark job that does the following:
# MAGIC   * tokenizes, removes stopwords and computes a word count on the ngram data
# MAGIC   * subsets the top 10,000 words (these are the terms we'll consider as potential synonyms)
# MAGIC   * subsets words 9,000-9,999 (this will be our 1,000 word basis vocabulary)    
# MAGIC   (to put it another way - of the top 10,000 words, the bottom 1,000 form the basis vocabulary)
# MAGIC   * saves the full 10K word list and the 1K basis vocabulary to file for use in `d`.  
# MAGIC   
# MAGIC   __NOTE:__ _to ensure consistency in results please use only the provided list of stopwords._  
# MAGIC   __NOTE:__ _as always, be sure to test your code on small files as you develop it._  
# MAGIC 
# MAGIC * __d) code:__ Write a spark job that builds co-occurrence stripes for the top 10K words in the ngram data using the basis vocabulary you developed in `part c`. This job/function, unlike others so far, should return an RDD (which we will then use in q8).

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Q7 Student Answers:
# MAGIC > __a)__ As we saw previously, choosing the least frequent words have issues of proper nouns or obscure words due to a small subset of documents. This makes it difficult to identify true synonyms. Choosing the most frequent words will get many meaningless stop words. Assuming stop words are filtered out, choosing most frequent words is likely to create results that are more generalizable (less overfit to our dataset). This is because the choosing words will have more chances to appear in the testing set, rather than the few words exist across disparate documents in the training set. Furthermore, the most frequently occurring generally will have more associated data, thus, the variance in the predictions will be lower.
# MAGIC 
# MAGIC > __b)__ Take random sample to be used as an unbiased esimator of fractional word frequency. For example, we can simply use the subsample; or we can apply the expected value by applying the fractional frequency to to the overall corpus.

# COMMAND ----------

# part c - provided stopwords (RUN THIS CELL AS IS)
STOPWORDS =  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
              'ourselves', 'you', 'your', 'yours', 'yourself', 
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 
              'her', 'hers', 'herself', 'it', 'its', 'itself', 
              'they', 'them', 'their', 'theirs', 'themselves', 
              'what', 'which', 'who', 'whom', 'this', 'that', 
              'these', 'those', 'am', 'is', 'are', 'was', 'were', 
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 
              'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
              'but', 'if', 'or', 'because', 'as', 'until', 'while', 
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 
              'between', 'into', 'through', 'during', 'before', 
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 
              'further', 'then', 'once', 'here', 'there', 'when', 
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
              'too', 'very', 'should', 'can', 'now', 'will', 'just', 
              'would', 'could', 'may', 'must', 'one', 'much', "it's",
              "can't", "won't", "don't", "shouldn't", "hasn't"]

# COMMAND ----------

# part c - spark job
def get_vocab(rdd, n_total, n_basis):
    vocab, basis = None, None
    ############# YOUR CODE HERE ###############
    def counter(line):
      line = line.split('\t')
      line[0] = line[0].lower().split(' ')
      for i in line[0]:
          yield (i,int(line[1]))

    mapped_rdd = rdd.flatMap(counter) \
                    .reduceByKey(lambda a,b: a+b) \
                    .filter(lambda x: not(x[0] in STOPWORDS))


    vocab = sc.parallelize(mapped_rdd.takeOrdered(n_total, key=lambda x: -x[1]))
    basis = sc.parallelize(vocab.takeOrdered(n_basis, key=lambda x: x[1])).map(lambda x: x[0]) \
                                                                          .collect()
    vocab = vocab.map(lambda x: x[0]) \
                 .collect()
    
    ############# (END) YOUR CODE ##############
    return vocab, basis

# COMMAND ----------

# part c - run your job (RUN THIS CELL AS IS)
start = time.time()
VOCAB, BASIS = get_vocab(dataRDD, 10000, 1000)
print("Wall time: {} seconds".format(time.time() - start))
# Expected wall time on w261_homeworks cluster: 33.64917516708374 seconds

# COMMAND ----------

dbutils.fs.put(hw3_path+"vocabulary.txt",str(VOCAB),True)
dbutils.fs.put(hw3_path+"basis.txt",str(BASIS),True)

# COMMAND ----------

# part d - spark job
def buildStripes(rdd, vocab, basis):
    stripesRDD = None
    ############# YOUR CODE HERE ###############
    vocab = set(vocab)
    basis = set(basis)

    def emitStripes(line):
      from collections import defaultdict
      line = set(line.split('\t')[0].lower().split(' ')).intersection(vocab)
      len_line = len(line)
      for i in line:
        h = defaultdict(int)
        for j in line.intersection(basis):
          if (i != j):
            h[j] += 0
          else:
            continue
        if len(h) > 0:
          yield (i,h)
        else:
          pass

    def sumStripes(stripe1, stripe2):
      for i in stripe2:
        stripe1[i] += 0
      return stripe1

    stripesRDD = rdd.flatMap(emitStripes) \
                    .reduceByKey(sumStripes) \
                    .map(lambda x: (x[0],set(x[1].keys())))
    
    ############# (END) YOUR CODE ##############
    return stripesRDD

# COMMAND ----------

# part d - run your systems test (RUN THIS CELL AS IS)
VOCAB, BASIS = get_vocab(testRDD, 10, 10)
testStripesRDD = buildStripes(testRDD, VOCAB, BASIS)
start = time.time()
print(testStripesRDD.collect())
print("Wall time: {} seconds".format(time.time() - start))
# Expected wall time on w261_homeworks cluster: 0.11356496810913086 seconds
# Expected results
'''
[('worst', {'times'}), ('best', {'times'}), ('foolishness', {'age'}), ('times', {'age', 'best', 'worst'}), ('age', {'wisdom', 'foolishness', 'times'}), ('wisdom', {'age'})]
'''

# COMMAND ----------

# part d - run your single file test (RUN THIS CELL AS IS)
VOCAB, BASIS = get_vocab(f1RDD, 10000, 1000)
f1StripesRDD = buildStripes(f1RDD, VOCAB, BASIS).cache()
start = time.time()
print(f1StripesRDD.top(5))
print("Wall time: {} seconds".format(time.time() - start))
# Expected wall time on w261_homeworks cluster: 1.3129394054412842 seconds
# Expected results
'''
[('zur', {'zur'}), ('zippor', {'balak'}), ('zedong', {'mao'}), ('zeal', {'infallibility'}), 
('youth', {'mould', 'constrained'})]
'''

# COMMAND ----------

# part d - run the full analysis and take a look at a few stripes (RUN THIS CELL AS IS)
VOCAB = ast.literal_eval(open(hw3_path_open+"vocabulary.txt", "r").read())
BASIS = ast.literal_eval(open(hw3_path_open+"basis.txt", "r").read())
stripesRDD = buildStripes(dataRDD, VOCAB, BASIS).cache()

start = time.time()
for wrd, stripe in stripesRDD.top(3):
    print(wrd)
    print(list(stripe))
    print('-------')
print("Wall time: {} seconds".format(time.time() - start))
# Expected Wall time on w261_homeworks cluster:  25.928858757019043 seconds
# Expected results
'''
zones
['remotest', 'adhesion', 'buffer', 'subdivided', 'environments', 'gaza', 'saturation', 'localities', 'uppermost', 'warmer', 'residential', 'parks']
-------
zone
['tribal', 'narrower', 'fibrous', 'saturation', 'originate', 'auxiliary', 'ie', 'buffer', 'transitional', 'turbulent', 'vomiting', 'americas', 'articular', 'poorly', 'intervening', 'officially', 'accumulate', 'assisting', 'flexor', 'traversed', 'uppermost', 'unusually', 'cartilage', 'inorganic', 'illuminated', 'glowing', 'contamination', 'trigger', 'defines', 'masculine', 'avoidance', 'cracks', 'southeastern', 'penis', 'residential', 'atlas', 'excitation', 'persia', 'diffuse', 'subdivided', 'alaska', 'guides', 'au', 'sandy', 'penetrating', 'parked']
-------
zinc
['ammonium', 'coating', 'pancreas', 'insoluble', "alzheimer's", 'diamond', 'radioactive', 'metallic', 'weighing', 'dysfunction', 'wasting', 'phosphorus', 'transcription', 'dipped', 'hydroxide', 'burns', 'leukemia', 'dietary']
-------
'''

# COMMAND ----------

# part d - save your full stripes to file for ease of retrival later... (OPTIONAL)
stripesRDD.saveAsTextFile(hw3_path+'stripes')

# COMMAND ----------

# MAGIC %md # Question 8: Synonym Detection
# MAGIC 
# MAGIC We're now ready to perform the main synonym detection analysis. In the tasks below you will compute cosine, jaccard, dice and overlap similarity measurements for each pair of words in our vocabulary and then sort your results to find the most similar pairs of words in this dataset. __`IMPORTANT:`__ When you get to the sorting step please __sort on cosine similarity__ only, so that we can ensure consistent results from student to student. 
# MAGIC 
# MAGIC Remember to test each step of your work with the small files before running your code on the full dataset. This is a computationally intense task: well designed code can be the difference between a 20min job and a 2hr job. __`NOTE:`__ _as you are designing your code you may want to review questions 3 and 4 where we modeled some of the key pieces of this analysis._
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In question 7 you wrote a function that would create word stripes for each `term` in our vocabulary. These word stripes are essentially an 'embedded representation' of the `term`'s meaning. What is the 'feature space' for this representation? (i.e. what are the features of our 1-hot encoded vectors?). What is the maximum length of a stripe?
# MAGIC 
# MAGIC * __b) short response:__ Remember that we are going to treat these stripes as 'documents' and perform similarity analysis on them. The first step is to emit postings which then get collected to form an 'inverted index.' How many rows will there be in our inverted index? Explain.
# MAGIC 
# MAGIC * __c) short response:__ In the demo from question 2, we were able to compute the cosine similarity directly from the stripes (we did this using their vector form, but could have used the list instead). So why do we need the inverted index? (__`HINT:`__ _see your answer to Q4a & Q4b_)
# MAGIC 
# MAGIC * __d) code:__ Write a spark job that does the following:
# MAGIC   * loops over the stripes from Q7 and emits postings for the `term` (_remember stripe = document_)
# MAGIC   * aggregates the postings to create an inverted index
# MAGIC   * loops over all pairs of `term`s that appear in the same inverted index and emits co-occurrence counts
# MAGIC   * aggregates co-occurrences
# MAGIC   * uses the counts (along with the accompanying information) to compute the cosine, jacard, dice and overlap similarity metrics for each pair of words in the vocabulary 
# MAGIC   * retrieve the top 20 and bottom 20 most/least similar pairs of words
# MAGIC   * also returned the cached sorted RDD for use in the next question  
# MAGIC   __`NOTE 1`:__ _Don't forget to include the stripe length when you are creating the postings & co-occurrence pairs. A composite key is the way to go here._  
# MAGIC   __`NOTE 2`:__ _Please make sure that your final results are sorted according to cosine similarity otherwise your results may not match the expected result & you will be marked wrong._
# MAGIC   
# MAGIC * __e) code:__ Comment on the quality of the "synonyms" your analysis comes up with. Do you notice anything odd about these pairs of words? Discuss at least one idea for how you might go about improving on the analysis.

# COMMAND ----------

# MAGIC %md ### Q8 Student Answers:
# MAGIC > __a)__ The feature space for this representation is the basis vocabulary (the entire potential space that can be put into a stripe).Each of the terms in the stripe is a 'feature'. The max length of a stripe is bounded by the basis vocabulary. In this case, we can have a maximum stripe length of 999 words.
# MAGIC 
# MAGIC > __b)__ There will be 1,000 rows in our inverted index. The inverted index would have 1,000 potential keys, as it inverts the index and the value of the stripe.
# MAGIC 
# MAGIC > __c)__ Inverted index can decearese the usage of storage significantly. As we can see, it is bounded by 1,000 rows. In previous questions, we've seen things take as long as 1.5 hours to run. With smaller dataset we should be able to improve the performance.
# MAGIC 
# MAGIC > __e)__ Few of these pairs are actually synonyms. Many of the synonym pairs have recurring words (e.g. 'first' is designated as a synonym for at least 5 other words under this model), and in general the top pairs consist of many of the same words. This leads us to believe that there may be frequent repeats of the same sentence or line or window in the n-gram corpus. One way to improve this analysis might be to eliminate n-gram duplicates, which would allow us to view word windows in truly different contexts, rather than being weighed more heavily by frequency of occurrence in the same context.

# COMMAND ----------

# helper function for pretty printing (RUN THIS CELL AS IS)
def displayOutput(lines):
    template = "{:25}|{:6}, {:7}, {:7}, {:5}"
    print(template.format("Pair", "Cosine", "Jaccard", "Overlap", "Dice"))
    for pair, scores in lines:
        scores = [round(s,4) for s in scores]
        print(template.format(pair, *scores))

# COMMAND ----------

# MAGIC %md __`TIP:`__ Feel free to define helper functions within the main function to help you organize your code. Readability is important! Eg:
# MAGIC ```
# MAGIC def similarityAnlysis(stripesRDD):
# MAGIC     """main docstring"""
# MAGIC     
# MAGIC     simScoresRDD, top_n, bottom_n = None, None, None
# MAGIC     
# MAGIC     ############ YOUR CODE HERE ###########
# MAGIC     def helper1():
# MAGIC         """helper docstring"""
# MAGIC         return x
# MAGIC         
# MAGIC     def helper2():
# MAGIC         """helper docstring"""
# MAGIC         return x
# MAGIC         
# MAGIC     # main spark job starts here
# MAGIC     
# MAGIC         ...etc
# MAGIC     ############ (END) YOUR CODE ###########
# MAGIC     return simScoresRDD, top_n, bottom_n
# MAGIC ```

# COMMAND ----------

# part d - write your spark job in the space provided
def similarityAnalysis(stripesRDD, n):
    """
    This function defines a Spark DAG to compute cosine, jaccard, 
    overlap and dice scores for each pair of words in the stripes
    provided. 
    
    Output: an RDD, a list of top n, a list of bottom n
    """
    simScoresRDD, top_n, bottom_n = None, None, None
    
    ############### YOUR CODE HERE ################
    def inversion(pair):
      doc, stripe = pair
      for w in stripe:
          yield (w, [(doc,len(stripe))])

    def makeCompositeKey(inverted_index):
        word, postings = inverted_index
        # taking advantage of symmetry, output only (a,b), but not (b,a)
        for subset in itertools.combinations(sorted(postings), 2):
            yield (str(subset), 1)

    def similarities(line):
      (doc1, n1), (doc2, n2) = ast.literal_eval(line[0])
      total = int(line[1])
      jaccard = total / float(int(n1) + int(n2) - total)
      dice = 2*jaccard / (jaccard+1)
      overlap = total / (min(int(n2),int(n1)))
      cosine = total / ((int(n1)**0.5)*(int(n2)**0.5))
      yield (doc1+" - "+doc2, (cosine, jaccard, overlap, dice))

    result = stripesRDD.flatMap(inversion) \
                       .reduceByKey(lambda x,y: x+y) \
                       .flatMap(makeCompositeKey) \
                       .reduceByKey(lambda x,y:  x+y) \
                       .flatMap(similarities)
  #                      .flatMap(coOccurrence) \
  #                      .reduceByKey(lambda a,b: a+b)

    top_n = result.takeOrdered(n, key = lambda x: -x[1][0])
    bottom_n = result.takeOrdered(n, key=lambda x: x[1][0])
    result = result.collect()

    ############### (END) YOUR CODE ##############
    return result, top_n, bottom_n

# COMMAND ----------

# part d - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
start = time.time()
testResult, top_n, bottom_n = similarityAnalysis(testStripesRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 0.5179581642150879 seconds

# COMMAND ----------

# part d - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
start = time.time()
f1Result, top_n, bottom_n = similarityAnalysis(f1StripesRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result  on w261_homeworks cluster:  Wall time: 1.066291332244873 seconds


# COMMAND ----------

# part d - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
start = time.time()
result, top_n, bottom_n = similarityAnalysis(stripesRDD, 20)
print("Wall time: {} seconds".format(time.time() - start))
# Expected result on w261_homeworks cluster: Wall time: 138.1922881603241 seconds

# COMMAND ----------

displayOutput(top_n)

# COMMAND ----------

displayOutput(bottom_n)

# COMMAND ----------

# MAGIC %md __Expected output f1RDD:__  
# MAGIC <table>
# MAGIC <th>MOST SIMILAR:</th>
# MAGIC <th>LEAST SIMILAR:</th>
# MAGIC <tr><td><pre>
# MAGIC Pair                     |Cosine, Jaccard, Overlap, Dice 
# MAGIC commentary - lady        |   1.0,     1.0,     1.0,   1.0
# MAGIC commentary - toes        |   1.0,     1.0,     1.0,   1.0
# MAGIC commentary - reply       |   1.0,     1.0,     1.0,   1.0
# MAGIC curious - tone           |   1.0,     1.0,     1.0,   1.0
# MAGIC curious - lady           |   1.0,     1.0,     1.0,   1.0
# MAGIC curious - owe            |   1.0,     1.0,     1.0,   1.0
# MAGIC lady - tone              |   1.0,     1.0,     1.0,   1.0
# MAGIC reply - tone             |   1.0,     1.0,     1.0,   1.0
# MAGIC lady - toes              |   1.0,     1.0,     1.0,   1.0
# MAGIC lady - reply             |   1.0,     1.0,     1.0,   1.0
# MAGIC </pre></td>
# MAGIC <td><pre>
# MAGIC 
# MAGIC Pair                     |Cosine, Jaccard, Overlap, Dice 
# MAGIC part - time              |0.0294,  0.0149,  0.0303, 0.0294
# MAGIC time - upon              |0.0314,  0.0159,  0.0345, 0.0312
# MAGIC time - two               |0.0314,  0.0159,  0.0345, 0.0312
# MAGIC made - time              |0.0325,  0.0164,   0.037, 0.0323
# MAGIC first - time             |0.0338,  0.0169,    0.04, 0.0333
# MAGIC new - time               |0.0352,  0.0175,  0.0435, 0.0345
# MAGIC part - us                |0.0355,  0.0179,  0.0417, 0.0351
# MAGIC little - part            |0.0355,  0.0179,  0.0417, 0.0351
# MAGIC made - two               |0.0357,  0.0182,   0.037, 0.0357
# MAGIC made - upon              |0.0357,  0.0182,   0.037, 0.0357
# MAGIC </pre></td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC __Expected output dataRDD:__  
# MAGIC <table>
# MAGIC <th>Most Similar</th>
# MAGIC <th>Least Similar</th>
# MAGIC <tr><td><pre>
# MAGIC Pair           |Cosine, Jaccard, Overlap, Dice 
# MAGIC first - time   |  0.89,  0.8012,  0.9149, 0.8897
# MAGIC time - well    |0.8895,   0.801,   0.892, 0.8895
# MAGIC great - time   | 0.875,  0.7757,   0.925, 0.8737
# MAGIC part - well    | 0.874,  0.7755,  0.9018, 0.8735
# MAGIC first - well   |0.8717,  0.7722,  0.8936, 0.8715
# MAGIC part - time    |0.8715,  0.7715,  0.9018, 0.871
# MAGIC time - upon    |0.8668,   0.763,  0.9152, 0.8656
# MAGIC made - time    | 0.866,  0.7619,  0.9109, 0.8649
# MAGIC made - well    |0.8601,  0.7531,  0.9022, 0.8592
# MAGIC time - way     |0.8587,  0.7487,  0.9259, 0.8563
# MAGIC great - well   |0.8526,  0.7412,  0.8988, 0.8514
# MAGIC time - two     |0.8517,  0.7389,  0.9094, 0.8498
# MAGIC first - great  |0.8497,  0.7381,  0.8738, 0.8493
# MAGIC first - part   |0.8471,  0.7348,  0.8527, 0.8471
# MAGIC great - upon   |0.8464,  0.7338,  0.8475, 0.8464
# MAGIC upon - well    |0.8444,   0.729,   0.889, 0.8433
# MAGIC new - time     |0.8426,   0.724,  0.9133, 0.8399
# MAGIC first - two    |0.8411,  0.7249,  0.8737, 0.8405
# MAGIC way - well     |0.8357,  0.7146,  0.8986, 0.8335
# MAGIC time - us      |0.8357,  0.7105,  0.9318, 0.8308
# MAGIC </pre></td>
# MAGIC <td><pre>
# MAGIC Pair                  |Cosine, Jaccard, Overlap, Dice 
# MAGIC region - write        |0.0067,  0.0032,  0.0085, 0.0065
# MAGIC relation - snow       |0.0067,  0.0026,  0.0141, 0.0052
# MAGIC cardiac - took        |0.0074,  0.0023,  0.0217, 0.0045
# MAGIC ever - tumor          |0.0076,   0.002,  0.0263, 0.004
# MAGIC came - tumor          |0.0076,   0.002,  0.0263, 0.004
# MAGIC let - therapy         |0.0076,   0.003,  0.0161, 0.0059
# MAGIC related - stay        |0.0078,  0.0036,  0.0116, 0.0072
# MAGIC factors - hear        |0.0078,  0.0039,  0.0094, 0.0077
# MAGIC implications - round  |0.0078,  0.0033,  0.0145, 0.0066
# MAGIC came - proteins       |0.0079,   0.002,  0.0286, 0.0041
# MAGIC population - window   |0.0079,  0.0039,    0.01, 0.0077
# MAGIC love - proportional   | 0.008,  0.0029,  0.0185, 0.0058
# MAGIC got - multiple        | 0.008,  0.0034,  0.0149, 0.0067
# MAGIC changes - fort        |0.0081,  0.0032,  0.0161, 0.0065
# MAGIC layer - wife          |0.0081,  0.0038,  0.0119, 0.0075
# MAGIC five - sympathy       |0.0081,  0.0034,  0.0149, 0.0068
# MAGIC arrival - essential   |0.0081,   0.004,  0.0093, 0.008
# MAGIC desert - function     |0.0081,  0.0031,  0.0175, 0.0062
# MAGIC fundamental - stood   |0.0081,  0.0038,  0.0115, 0.0077
# MAGIC patients - plain      |0.0081,   0.004,  0.0103, 0.0079
# MAGIC </pre></td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md ### Congratulations, you have completed HW3! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform

# COMMAND ----------


