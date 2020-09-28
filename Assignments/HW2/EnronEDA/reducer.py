#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
############ YOUR CODE HERE #########
    # Existing word
    if word == current_word:
        if is_spam == '1':
            spam_count += int(count)
        else:
            ham_count += int(count)
    # New word
    else: 
        if current_word !=None:
            if spam_count != 0:
                print(f'{current_word}\t{1}\t{spam_count}')
                spam_count = 0 # change spam_count to 0 avoid double counting
            if ham_count != 0:
                print(f'{current_word}\t{0}\t{ham_count}')
                ham_count = 0 # change ham_count to 0 avoid double counting
        
        if is_spam =='1':
            current_word, spam_count  = word, int(count) 
        else:
            current_word, ham_count  = word, int(count)

# print the last record! 
if is_spam == '1':
    print(f'{current_word}\t{1}\t{1}')
else: 
    print(f'{current_word}\t{0}\t{1}')

############ (END) YOUR CODE #########