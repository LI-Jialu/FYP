import numpy as np
import pandas as pd

def normalize(X):
    m,n = X.shape
    for j in range(n):
        features = X[:, j]
        mean = features.mean(0)
        std = features.std(0)
        if(std != 0):
            X[:, j] = (features - mean) / std
        else:
            X[:, j] = 0
    return X

def _split(array, interval): 
    rem = len(array)%interval
    inter_num = (len(array)-rem)/interval
    if rem != 0: 
        array = array[:-rem]
    splited = np.split(array,inter_num)
    return splited 
    
def split(array, interval):
    splited = _split(array,interval)
    return np.array([[*s[0][0:],*s[-1:][0]] for s in splited])

def interval_mean(array, interval): 
    temp = _split(array, interval)
    mean = np.array([t.mean(axis = 0) for t in temp])
    return mean

def interval_var(array, interval): 
    temp = _split(array, interval)
    var = np.array([t.var(axis = 0) for t in temp])
    return var

'''
a = np.array([[1,2,1,2], [3,4,3,4], [5,6,5,6],
                [7,8,7,8],[9,0,9,0],[11,12,11,12],
                [13,14,13,14],[15,16,15,16]])  
# print(split(a,3))
# print(interval_mean(a,3))
print(np.array([a_ele[0]-a_ele[-1] for a_ele in a]))
'''