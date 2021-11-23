import numpy as np

class interval_split: 
    def __init__(self, array, interval):
        self.array = array 
        self.interval = interval 

    def split(self, array,interval): 
        rem = len(array)%interval
        inter_num = (len(array)-rem)/interval
        if rem != 0: 
            array = array[:-rem]
        splited = np.split(array,inter_num)
        return splited 
        
    def beg_end(self, splited):
        return np.array([[s[0][0:],s[-1:][0]] for s in splited])
    
    def interval_mean(self, splited): 
        return np.mean(splited)
    
    def interval_var(self, splited): 
        return np.var(splited)



