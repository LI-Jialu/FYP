import datetime as datetime
import numpy as np 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from download_order_book import download_order_book as dob
from model_builder import model_builder, conditional_model_builder
import generate_pos as gp

##----------------------------------------------Parameters--------------------------------------------## 
interval = 500 #500, 1000, 2000
pcs = 5 # 2, 10, the number of targeted principal componenets
starttime_0 = datetime.datetime.now()
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


##---------------------------------Train Models without condition-------------------------------------##
mb = model_builder(pcs, interval, path)
tt_result = mb.build_model()
score_list = tt_result[:4]
report_list = tt_result[4:]

##-----------------------------------SVM Interval with PCA with condition ----------------------------## 
cmb = conditional_model_builder(pcs, interval, path)
tt_result = cmb.build_model()
pred_y_list = tt_result[:4]
conditional_score_list = tt_result[4:8]
conditional_report_list = tt_result[8:]

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
