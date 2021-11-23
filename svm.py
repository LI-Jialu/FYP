import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import split 

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

class svm_interval: 
    def __init__(self):
        pass

    def interval_feature(self, df, interval): 
        timestamps = np.array(df['timestamp'])
        N = timestamps.shape[0]

        # Feature Set V1
        f1 = df[['Pa1', 'Pa2', 'Pa3', 'Pa4', 'Pa5', 'Pa6', 'Pa7', 'Pa8', 'Pa9', 'Pa10',
                'Va1', 'Va2', 'Va3', 'Va4', 'Va5', 'Va6', 'Va7', 'Va8', 'Va9', 'Va10',
                'Pb1', 'Pb2', 'Pb3', 'Pb4', 'Pb5', 'Pb6', 'Pb7', 'Pb8', 'Pb9', 'Pb10',
                'Vb1', 'Vb2', 'Vb3', 'Vb4', 'Vb5', 'Vb6', 'Vb7', 'Vb8', 'Vb9', 'Vb10',]]
        f1 =split(np.array(f1),interval)


        # Feature Set V2
        temp1 = f1[:, 0:10] - f1[:, 20:30]
        temp2 = (f1[:, 0:10] + f1[:, 20:30]) * 0.5
        f2 = split(np.concatenate((temp1, temp2), axis = 1),interval)


        # Feature Set V3
        temp1 = (f1[:, 9] - f1[:, 0]).reshape(N, 1)
        temp2 = (f1[:, 20] - f1[:, 29]).reshape(N, 1)
        temp3 = abs(f1[:, 1:10] - f1[:, 0:9])
        temp4 = abs(f1[:, 21:30] - f1[:, 20:29])
        f3 = split(np.concatenate((temp1, temp2, temp3, temp4), axis = 1),interval)

        # Feature Set V4: mean prices and volumes
        temp1 = np.mean(f1[:, :10], axis = 1).reshape(N, 1)
        temp2 = np.mean(f1[:, 20:30], axis = 1).reshape(N, 1)
        temp3 = np.mean(f1[:, 10:20], axis = 1).reshape(N, 1)
        temp4 = np.mean(f1[:, 30:], axis = 1).reshape(N, 1)
        f4 = split(np.concatenate((temp1, temp2, temp3, temp4), axis = 1),interval)

        # Feature Set V5: accumulated differences
        temp1 = np.sum(f2[:, 0:10], axis = 1).reshape(N, 1)
        temp2 = np.sum(f1[:, 10:20] - f1[:, 30:40], axis = 1).reshape(N, 1)
        f5 = split(np.concatenate((temp1, temp2), axis = 1),interval)

        # Feature Set V6: Time interval for numbers of orders 
        f6 =split(np.array(df[['timestamp']]),interval)

        # Feature Set V7: Mean and variance 

        ## Price difference of b1, a1, b2, a2, 
        return f1, f2, f3, f4, f5, f6 

    

    def _X(self, f1, f2, f3, f4, f5,f6, N): 
    # Concatenate all features and normalize
        X = np.concatenate((f1, f2, f3, f4, f5), axis = 1)
        X = np.delete(X, 0, axis = 0)
        X = np.concatenate((X, f6), axis = 1)
        X = np.delete(X, N-2, axis = 0)
        X = normalize(X)

    def _y(self, f2, N): 
        N = N - 2
        # mid-price trend label
        y = np.zeros([N,], dtype = 'int')
        threshold = 0.1
        for i in range(N):
            y[i] = 0 if (abs(f2[i+2, 10] - f2[i+1, 10]) < threshold) else (1 if (f2[i+2, 10] > f2[i+1, 10]) else -1)


class svm_timepoint: 
    def __init__(self):
        pass

    def timpoint_feature(self, df):

        timestamps = np.array(df['timestamp'])
        N = timestamps.shape[0]

        # Feature Set V1
        f1 = df[['Pa1', 'Pa2', 'Pa3', 'Pa4', 'Pa5', 'Pa6', 'Pa7', 'Pa8', 'Pa9', 'Pa10',
                'Va1', 'Va2', 'Va3', 'Va4', 'Va5', 'Va6', 'Va7', 'Va8', 'Va9', 'Va10',
                'Pb1', 'Pb2', 'Pb3', 'Pb4', 'Pb5', 'Pb6', 'Pb7', 'Pb8', 'Pb9', 'Pb10',
                'Vb1', 'Vb2', 'Vb3', 'Vb4', 'Vb5', 'Vb6', 'Vb7', 'Vb8', 'Vb9', 'Vb10',]]
        f1 = np.array(f1)

        # Feature Set V2
        temp1 = f1[:, 0:10] - f1[:, 20:30]
        temp2 = (f1[:, 0:10] + f1[:, 20:30]) * 0.5
        f2 = np.concatenate((temp1, temp2), axis = 1)

        # Feature Set V3
        temp1 = (f1[:, 9] - f1[:, 0]).reshape(N, 1)
        temp2 = (f1[:, 20] - f1[:, 29]).reshape(N, 1)
        temp3 = abs(f1[:, 1:10] - f1[:, 0:9])
        temp4 = abs(f1[:, 21:30] - f1[:, 20:29])
        f3 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        # Feature Set V4: mean prices and volumes
        temp1 = np.mean(f1[:, :10], axis = 1).reshape(N, 1)
        temp2 = np.mean(f1[:, 20:30], axis = 1).reshape(N, 1)
        temp3 = np.mean(f1[:, 10:20], axis = 1).reshape(N, 1)
        temp4 = np.mean(f1[:, 30:], axis = 1).reshape(N, 1)
        f4 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        # Feature Set V5: accumulated differences
        temp1 = np.sum(f2[:, 0:10], axis = 1).reshape(N, 1)
        temp2 = np.sum(f1[:, 10:20] - f1[:, 30:40], axis = 1).reshape(N, 1)
        f5 = np.concatenate((temp1, temp2), axis = 1)

        # Feature Set V6: price and volume derivatives
        temp1 = f1[1:, 0:10] - f1[:-1, 0:10]
        temp2 = f1[1:, 20:30] - f1[:-1, 20:30]
        temp3 = f1[1:, 10:20] - f1[:-1, 10:20]
        temp4 = f1[1:, 30:40] - f1[:-1, 30:40]
        f6 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        return f1, f2, f3, f4, f5, f6 
    
    def _X(self, f1, f2, f3, f4, f5,f6, N): 
    # Concatenate all features and normalize
        X = np.concatenate((f1, f2, f3, f4, f5), axis = 1)
        X = np.delete(X, 0, axis = 0)
        X = np.concatenate((X, f6), axis = 1)
        X = np.delete(X, N-2, axis = 0)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
        return [X_train, X_test, y_train, y_test]

    def trian(self, X_train, y_train): 
        model = svm.SVC(kernel = 'rbf')
        model.fit(X_train, y_train)
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


    

    