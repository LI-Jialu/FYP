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
    
def split(array, window_num):
    splited = np.split(array, window_num)
    return np.array([[*s[0][0:],*s[-1:][0]] for s in splited])

def interval_mean(array, window_num): 
    temp = np.split(array, window_num)
    mean = np.array([t.mean(axis = 0) for t in temp]).reshape(-1,1)
    return mean

def interval_var(array, window_num): 
    temp = np.split(array, window_num)
    var = np.array([t.var(axis = 0) for t in temp]).reshape(-1,1)
    return var

'''
a = np.array([[1,2,1,2], [3,4,3,4], [5,6,5,6],
                [7,8,7,8],[9,0,9,0],[11,12,11,12],
                [13,14,13,14],[15,16,15,16]])  
# print(split(a,3))
# print(interval_mean(a,3))
print(np.array([a_ele[0]-a_ele[-1] for a_ele in a]))
'''