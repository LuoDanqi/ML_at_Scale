#!/usr/bin/env python
"""
This script reads word counts from STDIN and aggregates
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE (standalone):
    python aggregateCounts_v2.py < yourCountsFile.txt

Instructions:
    For Q7 - Your solution should not use a dictionary or store anything   
             other than a single total count - just print them as soon as  
             you've added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys


################# YOUR CODE HERE #################
# get the initial line and compare the next line
line1 = sys.stdin.readline()
key, val = line1.split()
val = int(val)

for line in sys.stdin:
    word, count = line.split()
    if word == key or word + 's' == key:
        val += int(count)
    else:
        # output when words are different
        print("{}\t{}".format(key, val))
        # replace the initial line
        key = word
        val = int(count)
# print the last line
print("{}\t{}".format(key, val))

################ (END) YOUR CODE #################
