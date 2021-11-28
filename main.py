import datetime as datetime
import numpy as np 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from download_order_book import download_order_book as dob
import download_price as dp 
from svm import svm_timepoint, svm_interval, train_test




# parameters 
interval = 50 #50, 100, 200
# the number of targeted principal componenets
pcs = 5 # 2, 10

starttime_0 = datetime.datetime.now()
tt = train_test()
##-------------------------------------------Data collection-----------------------------------------## 
DOB = dob()
DOB.download_order_book() 

raw_data_path = './raw_data' # folder to store raw data
tickers_list = ['BTC']
tf = '1m' # 1 minute 
DP = dp.download_price(tickers_list, tf, raw_data_path)
DP.get_raw()
DP.get_file('close')
DP.get_file('volume')
DP.combine_csv()

# Fear and Greedy index 
fng = np.array(pd.read_csv('./Data/fng_index.csv', header = 0)['fng_value'])

##---------------------------------Data loading & preprocessing--------------------------------------## 
# pd.set_option('display.max_columns', None)
# path = 'C:/tardis_dataset/bitmex'
path = './Data/Order_book'
date_list = ['2021-09-23', '2021-09-24', '2021-09-25', '2021-09-26', '2021-09-27', '2021-09-28', 
             '2021-09-29', '2021-09-30', '2021-10-01', '2021-10-02', '2021-10-03', '2021-10-04', 
             '2021-10-05', '2021-10-06', ]
day_num = len(date_list)
print(os.chdir('..'))
df = DOB.load_data(path, '2021-10-02')
# df = [DOB.load_data(path, date_list[i]) for i in range(day_num)]

##----------------------------------------SVM Single timepoint-------------------------------------------##
starttime_point = datetime.datetime.now()
svm_s = svm_timepoint(df[10]) # use the data of 2021-10-03
f1, f2, f3, f4, f5, f6 = svm_s.timpoint_feature()
X = svm_s.generate_X(f1, f2, f3, f4, f5, f6)
y = svm_s.generate_y(f2)
X_train, X_test, y_train, y_test = tt.train_test_split(X, y)
svm_model_1 = tt.trian(X_train, y_train)
pred_y_1 = tt.pred(svm_model_1, X_test, y_test)
tt.dump(svm_model_1)
endtime_point = datetime.datetime.now()
print('The SVM using one-timestamp datapoint running time:')
print((endtime_point - starttime_point).seconds)

##----------------------------------------SVM Interval (3 labels)---------------------------------------## 
starttime_interval = datetime.datetime.now()
svm_i = svm_interval(df[10], interval) # use the data of 2021-10-03
f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
X = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
y = svm_i.generate_y(f2, X.shape[0], 3)
X_train, X_test, y_train, y_test = tt.train_test_split(X, y)
svm_model_2 = tt.train(X_train, y_train)
pred_y_2 = tt.pred(svm_model_2, X_test, y_test)
tt.dump(svm_model_2)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window (3 labels) running time:')
print((endtime_interval - starttime_interval).seconds)

##----------------------------------------SVM Interval (5 labels)--------------------------------------## 
starttime_interval = datetime.datetime.now()
y = svm_i.generate_y(f2, X.shape[0], 5)
X_train, X_test, y_train, y_test = tt.train_test_split(X, y)
svm_model_3 = tt.train(X_train, y_train)
pred_y_3 = tt.pred(svm_model_3, X_test, y_test)
tt.dump(svm_model_3)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window (5 labels) running time:')
print((endtime_interval - starttime_interval).seconds)

##-------------------------------------------SVM Interval with PCA-------------------------------------## 
starttime_interval = datetime.datetime.now()
pca = PCA(n_components = pcs)
pca.fit(X)
X = pca.transform(X)
X_train, X_test, y_train, y_test = tt.train_test_split(X, y)
svm_model_4 = tt.train(X_train, y_train)
pred_y_4 = tt.pred(svm_model_4, X_test, y_test)
tt.dump(svm_model_4)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window (5 labels) and PCA running time:')
print((endtime_interval - starttime_interval).seconds)

##-----------------------------------SVM Interval with PCA with condition ----------------------------## 
X_train_greed = np.zeros((1,pcs), dtype = 'float64')
X_test_greed = np.zeros((1,pcs), dtype = 'float64')
y_train_greed = np.zeros((1,pcs), dtype = 'float64')
y_test_greed = np.zeros((1,pcs), dtype = 'float64')
X_train_fear = np.zeros((1,pcs), dtype = 'float64')
X_test_fear = np.zeros((1,pcs), dtype = 'float64')
y_train_fear = np.zeros((1,pcs), dtype = 'float64')
y_test_fear = np.zeros((1,pcs), dtype = 'float64')
starttime_interval = datetime.datetime.now()
for i in range(day_num):
    svm_i = svm_interval(df[i], interval)
    f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
    X = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
    y = svm_i.generate_y(f2, X.shape[0], 5)
    pca = PCA(n_components = pcs)
    pca.fit(X)
    X = pca.transform(X)
    X_train, X_test, y_train, y_test = tt.train_test_split(X, y)
    if (fng[i] < 50):
        np.concatenate((X_train_fear, X_train))
        np.concatenate((X_test_fear, X_test))
        np.concatenate((y_train_fear, y_train))
        np.concatenate((y_test_fear, y_test))
    elif (fng[i] >= 50):
        np.concatenate((X_train_greed, X_train))
        np.concatenate((X_test_greed, X_test))
        np.concatenate((y_train_greed, y_train))
        np.concatenate((y_test_greed, y_test))
X_train_greed = np.delete(X_train_greed, 0, axis = 0)
X_test_greed = np.delete(X_test_greed, 0, axis = 0)
y_train_greed = np.delete(y_train_greed, 0, axis = 0)
y_test_greed = np.delete(y_test_greed, 0, axis = 0)
X_train_fear = np.delete(X_train_fear, 0, axis = 0)
X_test_fear = np.delete(X_test_fear, 0, axis = 0)
y_train_fear = np.delete(y_train_fear, 0, axis = 0)
y_test_fear = np.delete(y_test_fear, 0, axis = 0)
svm_model_greed = tt.trian(X_train_greed, y_train_greed)
svm_model_fear = tt.train(X_train_fear, y_train_fear)
pred_y_greed = tt.train(svm_model_greed, X_test_greed, y_test_greed)
pred_y_fear = tt.train(svm_model_fear, X_test_fear, y_test_fear)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window and PCA and Condition (Greed or Fear) running time:')
print((endtime_interval - starttime_interval).seconds)


##-----------------------------------------------backtesting------------------------------------------## 

##-----------------------------------------------analysis------------------------------------------## 
# testing different time interval selection 
# testing different PCA dimension 
# plot accuracy plot comparison 
# plot the prediction on the stock graph 
# Can this model also work for ETH? 

##-------------------------------------------------End---------------------------------------------## 
endtime_0 = datetime.datetime.now()
print((endtime_0 - starttime_0).seconds)
print('The whole running time:')
