#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.

INPUT:
    partitionKey \t word \t class0_partialCount,class1_partialCount 
OUTPUT:
    word \t class0_totalCount,class1_totalCount, class0_probability,class1_probability 

Instructions:
    Again, you are free to design a solution however you see 
    fit as long as your final model meets our required format
    for the inference job we designed in Question 8. Please
    comment your code clearly and concisely.
    
    A few reminders: 
    1) Don't forget to emit Class Priors (with the right key).
    2) In python2: 3/4 = 0 and 3/float(4) = 0.75
"""
##################### YOUR CODE HERE ####################
import sys

# initialize aggregator
T_word_class0 = 0
T_word_class1 = 0
T_doc_class0 = 0
T_doc_class1 = 0

# initialize trackers
cur_word = None
cur_count_class0 = 0
cur_count_class1 = 0

# read input key-value pairs from standard input
for line in sys.stdin:
    # parse data
    pkeys, key, value = line.split('\t')                
    class0_count, class1_count = value.split(',')
    
    # tally counts from current key                 
    if key == cur_word:                             
        cur_count_class0 += int(class0_count)
        cur_count_class1 += int(class1_count)
    # OR ...                                        
    else:                                          
        # store word count total                    
        if cur_word == '!total_words':                    
            T_word_class0 = int(cur_count_class0)
            T_word_class1 = int(cur_count_class1)
            
        # store doc count total                    
        if cur_word == '!total_docs':                    
            T_doc_class0 = int(cur_count_class0)
            T_doc_class1 = int(cur_count_class1)
            
        # emit realtive frequency                   
        if cur_word and cur_word != '!total_words' and cur_word !='!total_docs':       
            print(f'{cur_word}\t{cur_count_class0},{cur_count_class1},{float(cur_count_class0/T_word_class0)},{float(cur_count_class1/T_word_class1)}') 
        # and start a new tally                     
        cur_word, cur_count_class0,cur_count_class1  = key, int(class0_count), int(class1_count)      
                                                    
## don't forget the last record!                    
print(f'{cur_word}\t{cur_count_class0},{cur_count_class1},{float(cur_count_class0/T_word_class0)},{float(cur_count_class1/T_word_class1)}') 

# Calculate the total values in the last reducer
if str(pkeys) == 'D':
    print(f'ClassPriors\t{T_doc_class0},{T_doc_class1},{T_doc_class0/T_doc_class0+T_doc_class1},{T_doc_class1/T_doc_class0+T_doc_class1}')



































##################### (END) CODE HERE ####################