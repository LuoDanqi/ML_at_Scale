#!/usr/bin/env python
"""
Reducer to calculate precision and recall as part
of the inference phase of Naive Bayes.
INPUT:
    ID \t true_class \t P(ham|doc) \t P(spam|doc) \t predicted_class
OUTPUT:
    precision \t ##
    recall \t ##
    accuracy \t ##
    F-score \t ##
         
Instructions:
    Complete the missing code to compute these^ four
    evaluation measures for our classification task.
    
    Note: if you have no True Positives you will not 
    be able to compute the F1 score (and maybe not 
    precision/recall). Your code should handle this 
    case appropriately feel free to interpret the 
    "output format" above as a rough suggestion. It
    may be helpful to also print the counts for true
    positives, false positives, etc.
"""
import sys

# initialize counters
FP = 0.0 # false positives
FN = 0.0 # false negatives
TP = 0.0 # true positives
TN = 0.0 # true negatives

# read from STDIN
for line in sys.stdin:
    # parse input
    docID, class_, pHam, pSpam, pred = line.split()
    # emit classification results first
    print(line[:-2], class_ == pred)
    
    # then compute evaluation stats
#################### YOUR CODE HERE ###################

    if class_ == pred and pred == '1':
        TP +=1
    elif class_ == pred and pred == '0':
        TN +=1
    elif class_ != pred and pred == '1':
        FP +=1
    else:
        FN +=1
        
if TP == 0:
    TP = -1 # Assign a negative value for flag
else:
    # Get the values
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    Fscore = 2*precision*recall/(precision+recall)

# Print out 
print(f'# Documents:\t {(TP+TN+FP+FN)}')
print(f'True Positives:\t {TP}')
print(f'True Negatives:\t {TN}')
print(f'False Positives:\t {FP}')
print(f'False Negatives:\t {FN}')
print(f'accuracy\t{accuracy}')
print(f'precision\t{precision}')
print(f'recall\t{recall}')
print(f'F-score\t{Fscore}')
















#################### (END) YOUR CODE ###################
    