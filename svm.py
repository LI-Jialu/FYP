import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from interval_split import split, normalize
from interval_split import interval_mean as imean 
from interval_split import interval_var as ivar 
from sklearn.utils import class_weight


class svm_interval: 
    def __init__(self, df, interval):
        self.df = df 
        self.interval = interval 

    def interval_feature(self): 
        timestamps = np.array(self.df['timestamp'])
        N = timestamps.shape[0]
        ##### modify N here????

        # Feature Set V1   
        f1_temp = self.df[['Pa1', 'Pa2', 'Pa3', 'Pa4', 'Pa5', 
                            'Va1', 'Va2', 'Va3', 'Va4', 'Va5', 
                            'Pb1', 'Pb2', 'Pb3', 'Pb4', 'Pb5', 
                            'Vb1', 'Vb2', 'Vb3', 'Vb4', 'Vb5',]]
        f1_temp = np.array(f1_temp)
        f1 =split(f1_temp,self.interval)

        # Feature Set V2: price difference and mid-price 
        temp1 = f1_temp[:, 0:5] - f1_temp[:, 10:15]
        temp2 = (f1_temp[:, 0:5] + f1_temp[:, 10:15]) * 0.5
        f2_temp = np.concatenate((temp1, temp2), axis = 1)
        f2 = split(f2_temp,self.interval)

        # Feature Set V3
        temp1 = (f1_temp[:, 4] - f1_temp[:, 0]).reshape(-1, 1)
        temp2 = (f1_temp[:, 10] - f1_temp[:, 14]).reshape(-1, 1)
        temp3 = abs(f1_temp[:, 1:5] - f1_temp[:, 0:4])
        temp4 = abs(f1_temp[:, 11:15] - f1_temp[:, 10:14])
        f3 = split(np.concatenate((temp1, temp2, temp3, temp4), axis = 1),self.interval)

        # Feature Set V4: mean prices and volumes
        temp1 = np.mean(f1_temp[:, :5], axis = 1).reshape(-1, 1)
        temp2 = np.mean(f1_temp[:, 10:15], axis = 1).reshape(-1, 1)
        temp3 = np.mean(f1_temp[:, 5:10], axis = 1).reshape(-1, 1)
        temp4 = np.mean(f1_temp[:, 15:], axis = 1).reshape(-1, 1)
        f4 = split(np.concatenate((temp1, temp2, temp3, temp4), axis = 1),self.interval)

        # Feature Set V5: accumulated differences
        temp1 = np.sum(f2_temp[:, 0:5], axis = 1).reshape(-1, 1)
        temp2 = np.sum(f1_temp[:, 5:10] - f1_temp[:, 15:20], axis = 1).reshape(-1, 1)
        f5 = split(np.concatenate((temp1, temp2), axis = 1),self.interval)

        # Feature Set V6: Time interval for numbers of orders 
        f6_temp =split(np.array(self.df[['timestamp']]),self.interval)
        f6 = np.array([f[-1]-f[0] for f in f6_temp]).reshape(-1,1)

        # Feature Set V7: Mean and variance of bids and asks 
        mean = imean(f1_temp, self.interval)
        var = ivar(f1_temp, self.interval)
        f7 = np.concatenate((mean, var), axis = 1)
        
        # Feature Set V8: Price and Volume derivatives
        f8 = (f1[:, 20:40] - f1[:, 0:20]) / f6
    
        return f1, f2, f3, f4, f5, f6, f7, f8


    def generate_X(self, f1, f2, f3, f4, f5, f6, f7, f8): 
    # Concatenate all features and normalize
        X = np.concatenate((f1, f2, f3, f4, f5, f6, f7, f8), axis = 1)
        X = np.delete(X, -1, axis = 0)
        X = normalize(X)
        return X

    def generate_y(self, f2, N, label_num = 3): 
        # mid-price trend label
        y = np.zeros([N-1,], dtype = 'int')
        threshold1 = 0.1
        threshold2 = 2.0
        if(label_num == 3):
            for i in range(N-1):
                y[i] = 0 if (abs(f2[i+1, 30] - f2[i, 30]) < threshold1) else (1 if (f2[i+1, 30] > f2[i, 30]) else -1)
        elif(label_num == 5):
            for i in range(N-1):
                difference = f2[i+1, 30] - f2[i, 30]
                if(abs(difference) < threshold1):
                    y[i] = 0
                elif(abs(difference) < threshold2):
                    y[i] = 1 if difference > 0 else -1
                else:
                    y[i] = 2 if difference > 0 else -2
        return y 

class svm_timepoint: 
    def __init__(self, df):
        self.df = df 
        
    def timpoint_feature(self):
        timestamps = np.array(self.df['timestamp'])
        N = timestamps.shape[0]

        # Feature Set V1
        f1 = self.df[['Pa1', 'Pa2', 'Pa3', 'Pa4', 'Pa5', 
                    'Va1', 'Va2', 'Va3', 'Va4', 'Va5', 
                    'Pb1', 'Pb2', 'Pb3', 'Pb4', 'Pb5', 
                    'Vb1', 'Vb2', 'Vb3', 'Vb4', 'Vb5', ]]
        f1 = np.array(f1)

        # Feature Set V2
        temp1 = f1[:, 0:5] - f1[:, 10:15]
        temp2 = (f1[:, 0:5] + f1[:, 10:15]) * 0.5
        f2 = np.concatenate((temp1, temp2), axis = 1)

        # Feature Set V3
        temp1 = (f1[:, 4] - f1[:, 0]).reshape(-1, 1)
        temp2 = (f1[:, 10] - f1[:, 14]).reshape(-1, 1)
        temp3 = abs(f1[:, 1:5] - f1[:, 0:4])
        temp4 = abs(f1[:, 11:15] - f1[:, 10:14])
        f3 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        # Feature Set V4: mean prices and volumes
        temp1 = np.mean(f1[:, :5], axis = 1).reshape(-1, 1)
        temp2 = np.mean(f1[:, 10:15], axis = 1).reshape(-1, 1)
        temp3 = np.mean(f1[:, 5:10], axis = 1).reshape(-1, 1)
        temp4 = np.mean(f1[:, 15:], axis = 1).reshape(-1, 1)
        f4 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        # Feature Set V5: accumulated differences
        temp1 = np.sum(f2[:, 0:5], axis = 1).reshape(-1, 1)
        temp2 = np.sum(f1[:, 5:10] - f1[:, 15:20], axis = 1).reshape(-1, 1)
        f5 = np.concatenate((temp1, temp2), axis = 1)

        # Feature Set V6: price and volume derivatives
        temp1 = f1[1:, 0:5] - f1[:-1, 0:5]
        temp2 = f1[1:, 10:15] - f1[:-1, 10:15]
        temp3 = f1[1:, 5:10] - f1[:-1, 5:10]
        temp4 = f1[1:, 15:20] - f1[:-1, 15:20]
        f6 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        return f1, f2, f3, f4, f5, f6 
    
    def _X(self, f1, f2, f3, f4, f5, f6): 
    # Concatenate all features and normalize
        X = np.concatenate((f1, f2, f3, f4, f5), axis = 1)
        X = np.delete(X, 0, axis = 0)
        X = np.concatenate((X, f6), axis = 1)
        X = np.delete(X, -2, axis = 0)
        X = normalize(X)
        return X

    def _y(self, f2, N): 
        N = N - 2
        # mid-price trend label
        y = np.zeros([N,], dtype = 'int')
        threshold = 0.1
        for i in range(N):
            y[i] = 0 if (abs(f2[i+2, 10] - f2[i+1, 10]) < threshold) else (1 if (f2[i+2, 10] > f2[i+1, 10]) else -1)
        return y 

class train_test: 

    def __init__(self):
        pass    

    def train_test_split(self, X, y): 
        # train-test split
        return train_test_split(X, y, test_size = 0.2, stratify = y)

    def trian(self, X_train, y_train): 
        model = svm.LinearSVC()
        model.fit(X_train, y_train, class_weight = 'balanced')
        return model 

    def dump(self, model): 
        try:
            fp=open('model_SVC_RBF.pydata','wb')
            pickle.dump(model,fp)
            fp.close()
        except Exception as e:
            print(e)
        
    def pred(self, model, X_test, y_test):
        pred = model.predict(X_test)
        ap = model.score(X_test, y_test)
        print(ap)
        return pred 


    

    