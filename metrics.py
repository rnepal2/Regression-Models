import math
import numpy as np

# If we use this function with test data: 
def accuracy_score(y_true, y_pred):
    '''
    parameters:
        y_true: acutual target array
        y_pred: predicted target array
    '''
    assert len(y_true) == len(y_pred)
    
    if len(y_true) == 0: raise ValueError('Empty y_true!')
    
    correct = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1 == y2: correct += 1
    return correct/len(y_true)
    
# Generalization error = 1 - accuracy_score


# 8. 
# precision
def precision_score(y_true, y_pred):
    '''
    parameters:
        y: actual target array (first input array should be actual target)
        y_: predicted target array
    
    precision_score = TP/(TP + FP) {TP: True Positive, FP: False Positive}
    Note that: this precision_score is valid only for binary classification.
    '''
    # some confirmations and coversion if required
    if len(np.unique(y_true)) != 2:
        raise ValueError('the target has more than two classes!')
    
    if min(y_true)!=0 and min(y_pred)!=0 and max(y_true)!=1 and max(y_pred)!=1:
        raise ValueError('target can only be either 0 or 1!')
    
    tp = 0; fp = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1==1 and y2==1: tp += 1
        if y1==0 and y2==1: fp += 1
    if tp+fp==0: raise ValueError('TP+FP=0: precision cannot be defined.')
    return tp/(tp+fp)
        
# recall
def recall_score(y_true, y_pred):
    '''
    parameters:
        y_true: actual target array (first input array should be actual target)
        y_pred: predicted target array
    
    recall = TP/(TP + FN) {TP: True Positive, FN: False Negative}
    Note that: this recall is valid only for binary classification.
    '''
    # some confirmations and coversion if required
    if len(np.unique(y_true)) != 2:
        raise ValueError('the target has more than two classes!')
    
    if min(y_true)!=0 and min(y_pred)!=0 and max(y_true)!=1 and max(y_pred)!=1:
        raise ValueError('target can be only 0 and 1!')
    
    tp = 0; fn = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1==1 and y2==1: tp += 1
        if y1==1 and y2==0: fn += 1
    if tp+fn==0: raise ValueError('TP+FN=0: precision cannot be defined.')
    return tp/(tp+fn)

# F1-score
def f1_score(y_true, y_pred):
    '''
    parameters:
        y_true: actual target array (first input array should be actual target)
        y_pred: predicted target array
    
    f1_score = 2 precision*recall/(precision + recall) 
    or f1_score = harmonic mean of precision and recall
    Note that: this f1_score is valid only for binary classification.
    '''
    precision, recall = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    return 2*precision*recall/(precision + recall)

# confusion matrix
# returns a confusion_matrix for n-classifier 
# works for binary or higher number of classes
def confusion_matrix(y_true, y_pred):  
    '''
    parameters:
        y_true: actual target array
        y_pred: predicted target array
    '''
    # confirm input sizes
    assert len(y_true) == len(y_pred)
    
    # number of classes
    length = len(np.unique(y_true))
    decisions = dict()
    for y1, y2 in zip(y_true, y_pred):
        
        if str(y1) in decisions.keys():
            decisions[str(y1)].append(y2)
        else:
            decisions[str(y1)] = [y2]

    # cross checking if some class are missing
    max_calss = max(max(y_true), max(y_pred))
    length = max_calss + 1
            
    # sorting and counting numbers in a new nested dictionary 
    nested_decisions = dict()
    for key in sorted(decisions.keys()):
        values = decisions[key]
        inner_dict = dict()
        # initialize inner_dict
        for value in range(length):
            inner_dict[str(value)] = 0
        # counting values
        for value in values:
            inner_dict[str(value)] += 1
        nested_decisions[key] = inner_dict
        
    # creating confusion matrix
    # initialize the confusion matrix
    confusion_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            try:
                confusion_matrix[i][j] = nested_decisions[str(i)][str(j)]
            except:
                confusion_matrix[i][j] = 0
    return confusion_matrix





