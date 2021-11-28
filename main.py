import datetime as datetime
import numpy as np 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from download_order_book import download_order_book as dob
from svm import svm_timepoint, svm_interval, train_test
import generate_pos as gp

##----------------------------------------------Parameters--------------------------------------------## 
interval = 500 #500, 1000, 2000
pcs = 5 # 2, 10, the number of targeted principal componenets
starttime_0 = datetime.datetime.now()
tt = train_test()
##-------------------------------------------Data collection-----------------------------------------## 
DOB = dob()
# DOB.download_derivative_ticker() 
DOB.download_order_book() 
# Fear and Greedy index 
fng = np.array(pd.read_csv('./Data/fng_index.csv', header = 0)['fng_value'])

##---------------------------------Data loading & preprocessing--------------------------------------## 
# pd.set_option('display.max_columns', None)
# path = 'C:/tardis_dataset/bitmex'
path = './Data/Order_book'
print(os.chdir('..'))
df_train = DOB.load_data(path, '2021-10-02')
df_test = DOB.load_data(path, '2021-10-04')
### ?? How about others 

##----------------------------------------SVM Single timepoint-------------------------------------------##
starttime_point = datetime.datetime.now()
svm_s = svm_timepoint(df_train) # use the data of 2021-10-02 to train
f1, f2, f3, f4, f5, f6 = svm_s.timpoint_feature()
X_train = svm_s.generate_X(f1, f2, f3, f4, f5, f6)
y_train = svm_s.generate_y(f2)
svm_model_1 = tt.trian(X_train, y_train)

svm_s = svm_timepoint(df_test) # use the data of 2021-10-04 to test
f1, f2, f3, f4, f5, f6 = svm_s.timpoint_feature()
X_test = svm_s.generate_X(f1, f2, f3, f4, f5, f6)
y_test = svm_s.generate_y(f2)
pred_y_1 = tt.pred(svm_model_1, X_test, y_test)
timestamp_array = svm_s.get_timestamp_array()[1:-1]
pred_y_1 = tt.attach_timestamp(pred_y_1, timestamp_array)
tt.dump(svm_model_1)
endtime_point = datetime.datetime.now()
print('The SVM using one-timestamp datapoint running time:')
print((endtime_point - starttime_point).seconds)

##----------------------------------------SVM Interval (3 labels)---------------------------------------## 
starttime_interval = datetime.datetime.now()
svm_i_1 = svm_interval(df_train, interval) # use the data of 2021-10-02 to train
f1_1, f2_1, f3_1, f4_1, f5_1, f6_1, f7_1, f8_1 = svm_i_1.interval_feature()
X_train = svm_i_1.generate_X(f1_1, f2_1, f3_1, f4_1, f5_1, f6_1, f7_1, f8_1)
y_train = svm_i_1.generate_y(f2_1, f2_1.shape[0], 3)
svm_model_2 = tt.train(X_train, y_train)

svm_i_2 = svm_interval(df_test, interval) # use the data of 2021-10-04 to test
f1_2, f2_2, f3_2, f4_2, f5_2, f6_2, f7_2, f8_2 = svm_i_2.interval_feature()
X_test = svm_i_2.generate_X(f1_2, f2_2, f3_2, f4_2, f5_2, f6_2, f7_2, f8_2)
y_test = svm_i_2.generate_y(f2_2, f2_2.shape[0], 3)
pred_y_2 = tt.pred(svm_model_2, X_test, y_test)
timestamp_array = svm_i_2.get_timestamp_array()[:-1]
pred_y_2 = tt.attach_timestamp(pred_y_2, timestamp_array)
tt.dump(svm_model_2)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window (3 labels) running time:')
print((endtime_interval - starttime_interval).seconds)

##----------------------------------------SVM Interval (5 labels)--------------------------------------## 
starttime_interval = datetime.datetime.now()
y_train = svm_i_1.generate_y(f2_1, f2_1.shape[0], 5)
y_test = svm_i_2.generate_y(f2_2, f2_2.shape[0], 5)
svm_model_3 = tt.train(X_train, y_train)
pred_y_3 = tt.pred(svm_model_3, X_test, y_test)
pred_y_3 = tt.attach_timestamp(pred_y_3, timestamp_array)
tt.dump(svm_model_3)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window (5 labels) running time:')
print((endtime_interval - starttime_interval).seconds)

##-------------------------------------------SVM Interval with PCA-------------------------------------## 
starttime_interval = datetime.datetime.now()
pca = PCA(n_components = pcs)
pca.fit(X_train)
X_train = pca.transform(X_train)
pca = PCA(n_components = pcs)
pca.fit(X_test)
X_test = pca.transform(X_test)
svm_model_4 = tt.train(X_train, y_train)
pred_y_4 = tt.pred(svm_model_4, X_test, y_test)
pred_y_4 = tt.attach_timestamp(pred_y_4, timestamp_array)
tt.dump(svm_model_4)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window (5 labels) and PCA running time:')
print((endtime_interval - starttime_interval).seconds)

##-----------------------------------SVM Interval with PCA with condition ----------------------------## 
date_list = ['2021-09-29', '2021-09-30', '2021-09-31', 
             '2021-10-02', '2021-10-04', '2021-10-06', 
             '2021-10-28', '2021-10-05', '2021-10-03', ]
df = [DOB.load_data(path, date_list[i]) for i in range(9)]

X_train_greed = np.zeros((1,pcs), dtype = 'float64')
y_train_greed = np.zeros((1,pcs), dtype = 'float64')
X_train_fear = np.zeros((1,pcs), dtype = 'float64')
y_train_fear = np.zeros((1,pcs), dtype = 'float64')
starttime_interval = datetime.datetime.now()
for i in range(3):
    svm_i = svm_interval(df[i], interval)
    f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
    X_train = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
    y_train = svm_i.generate_y(f2, f2.shape[0], 5)
    pca = PCA(n_components = pcs)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    np.concatenate((X_train_fear, X_train))
    np.concatenate((y_train_fear, y_train))
for i in range(3,6):
    svm_i = svm_interval(df[i], interval)
    f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
    X_train = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
    y_train = svm_i.generate_y(f2, f2.shape[0], 5)
    pca = PCA(n_components = pcs)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    np.concatenate((X_train_greed, X_train))
    np.concatenate((y_train_greed, y_train))

svm_i = svm_interval(df[6], interval)
f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
X_test_fear = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
y_test_fear = svm_i.generate_y(f2, f2.shape[0], 5)
pca = PCA(n_components = pcs)
pca.fit(X_test_fear)
X_test_fear = pca.transform(X_test_fear)
timestamp_fear = svm_i.get_timestamp_array()[:-1]

svm_i = svm_interval(df[7], interval)
f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
X_test_greed = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
y_test_greed = svm_i.generate_y(f2, f2.shape[0], 5)
pca = PCA(n_components = pcs)
pca.fit(X_test_greed)
X_test_greed = pca.transform(X_test_greed)
timestamp_greed = svm_i.get_timestamp_array()[:-1]

svm_i = svm_interval(df[8], interval)
f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
X_test_neutral = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
y_test_neutral = svm_i.generate_y(f2, f2.shape[0], 5)
pca = PCA(n_components = pcs)
pca.fit(X_test_neutral)
X_test_neutral = pca.transform(X_test_neutral)
timestamp_neutral = svm_i.get_timestamp_array()[:-1]

X_train_greed = np.delete(X_train_greed, 0, axis = 0)
y_train_greed = np.delete(y_train_greed, 0, axis = 0)
X_train_fear = np.delete(X_train_fear, 0, axis = 0)
y_train_fear = np.delete(y_train_fear, 0, axis = 0)

svm_model_greed = tt.trian(X_train_greed, y_train_greed)
svm_model_fear = tt.train(X_train_fear, y_train_fear)

pred_y_greed = tt.train(svm_model_greed, X_test_greed, y_test_greed)
pred_y_fear = tt.train(svm_model_fear, X_test_fear, y_test_fear)
pred_y_neutral_fear = tt.train(svm_model_fear, X_test_neutral, y_test_neutral)
pred_y_neutral_greed = tt.train(svm_model_greed, X_test_neutral, y_test_neutral)
pred_y_greed = tt.attach_timestamp(pred_y_greed, timestamp_greed)
pred_y_fear = tt.attach_timestamp(pred_y_fear, timestamp_fear)
pred_y_neutral_fear = tt.attach_timestamp(pred_y_neutral_fear, timestamp_neutral)
pred_y_neutral_greed = tt.attach_timestamp(pred_y_neutral_greed, timestamp_neutral)

endtime_interval = datetime.datetime.now()
print('The SVM using sliding window and PCA and Condition (Greed or Fear) running time:')
print((endtime_interval - starttime_interval).seconds)

##---------------------------------------from prediction to position--------------------------------## 
GP = gp()
# For Model 1: 
pos = GP.kelly_criteria(model_1_pre_acc_1, model_1_pre_acc_m1, threshold_1, threshold_m1)
GP.three_label_pos(in_file,out_file,pos)
# For Model 2: 

##-----------------------------------------------backtesting----------------------------------------## 

for pos_file in pos_files: 
    backtest(pos_file)

##-----------------------------------------------analysis------------------------------------------## 
# testing different time interval selection 
# testing different PCA dimension 
# plot accuracy plot comparison 
# plot the prediction on the stock graph 
# Can this model also work for ETH? 
# View Pyfolio results

##-------------------------------------------------End---------------------------------------------## 
endtime_0 = datetime.datetime.now()
print((endtime_0 - starttime_0).seconds)
print('The whole running time:')
