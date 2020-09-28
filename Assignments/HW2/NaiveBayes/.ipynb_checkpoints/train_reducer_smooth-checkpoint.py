#!/usr/bin/env python
"""
INPUT:
    partitionKey \t word \t class0_partialCount,class1_partialCount 
OUTPUT:
    word \t class0_totalCount,class1_totalCount, class0_probability,class1_probability 

"""
import sys                                                  
import numpy as np 

#################### YOUR CODE HERE ###################

# initialize aggregator
T_word_class0 = 0
T_word_class1 = 0
T_doc_class0 = 0
T_doc_class1 = 0

# initialize trackers
cur_word = None
cur_count_class0 = 0
cur_count_class1 = 0
Vocab = 0

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
        
        if cur_word == '!uniq_words':
            Vocab = int(cur_count_class0)

            
        # emit realtive frequency                   
        if cur_word and cur_word != '!total_words' and cur_word !='!total_docs' and cur_word!='!uniq_words':          
            print(f'{cur_word}\t{cur_count_class0},{cur_count_class1},{float((cur_count_class0+1)/(T_word_class0 + Vocab))},{float((cur_count_class1+1)/(T_word_class1+Vocab))}') 
        # and start a new tally                     
        cur_word, cur_count_class0,cur_count_class1  = key, int(class0_count), int(class1_count)      
                                                    
## don't forget the last record! 
print(f'{cur_word}\t{cur_count_class0},{cur_count_class1},{float((cur_count_class0+1)/(T_word_class0 + Vocab))},{float((cur_count_class1+1)/(T_word_class1+Vocab))}')

# Calculate the total values in the last reducer
if str(pkeys) == 'D':
    print(f'ClassPriors\t{T_doc_class0},{T_doc_class1},{T_doc_class0/T_doc_class0+T_doc_class1},{T_doc_class1/T_doc_class0+T_doc_class1}')
































#################### (END) YOUR CODE ###################