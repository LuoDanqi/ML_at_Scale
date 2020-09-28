#!/usr/bin/env python
"""
Mapper reads in text documents and emits word counts by class.
INPUT:                                                    
    DocID \t true_class \t subject \t body                
OUTPUT:                                                   
    partitionKey \t word \t class0_partialCount,class1_partialCount       
    

Instructions:
    You know what this script should do, go for it!
    (As a favor to the graders, please comment your code clearly!)
    
    A few reminders:
    1) To make sure your results match ours please be sure
       to use the same tokenizing that we have provided in
       all the other jobs:
         words = re.findall(r'[a-z]+', text-to-tokenize.lower())
         
    2) Don't forget to handle the various "totals" that you need
       for your conditional probabilities and class priors.
       
Partitioning:
    In order to send the totals to each reducer, we need to implement
    a custom partitioning strategy.
    
    We will generate a list of keys based on the number of reduce tasks 
    that we read in from the environment configuration of our job.
    
    We'll prepend the partition key by hashing the word and selecting the
    appropriate key from our list. This will end up partitioning our data
    as if we'd used the word as the partition key - that's how it worked
    for the single reducer implementation. This is not necessarily "good",
    as our data could be very skewed. However, in practice, for this
    exercise it works well. The next step would be to generate a file of
    partition split points based on the distribution as we've seen in 
    previous exercises.
    
    Now that we have a list of partition keys, we can send the totals to 
    each reducer by prepending each of the keys to each total.
       
"""

import re                                                   
import sys                                                  
import numpy as np      

from operator import itemgetter
import os

#################### YOUR CODE HERE ###################

# Set 4 Partitions A,B,C,D by assigning alphabetically
def getPartitionKey(word):
    if word[0] < "g":
        return 'A'
    elif word[0] < "n":
        return 'B'
    elif word[0] < "t":
        return 'C'
    else:
        return 'D'
    
# Initialize local aggreagator of total word count and doc count
T_word_class0 = 0
T_word_class1 = 0
T_doc_class0 = 0
T_doc_class1 = 0
 
# initialize unique word count dictionary
Unique = {}

# Read from input
for line in sys.stdin:
    # parse input
    docID, _class, subject, body = line.lower().split('\t')
    # tokenize
    words = re.findall(r'[a-z]+', subject + ' ' + body)
    
    # get the total doc count
    if _class == '0':
        T_doc_class0 += 1
    else:
        T_doc_class1 += 1
        
    # get the word count and total count
    for word in words:
        # get unique key dictionary
        Unique[word] = Unique.get(word, 0) + 1
        
        if _class == '0':
            T_word_class0 += 1
            pkey = getPartitionKey(word)
            print(f'{pkey}\t{word}\t{1},{0}')
        else:
            T_word_class1 += 1
            pkey = getPartitionKey(word)
            print(f'{pkey}\t{word}\t{0},{1}')

for pkey in ['A','B','C','D']:                 
    print(f'{pkey}\t!total_words\t{T_word_class0}, {T_word_class1}') 
    print(f'{pkey}\t!total_docs\t{T_doc_class0}, {T_doc_class1}') 
    # Also output the unique words
    print(f'{pkey}\t!uniq_words\t{len(Unique.keys())}, {len(Unique.keys())}')

























#################### (END) YOUR CODE ###################