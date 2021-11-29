import datetime as datetime
import numpy as np 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from download_order_book import download_order_book as dob
from model_builder import model_builder, conditional_model_builder
import generate_pos as gp
from backtesting import backtesting

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
threshold1 = 5e-04 
threshold2 = 1e-03
# For Model 1: greed, backtest in greed 
pos1 = GP.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 = GP.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
GP.five_label_pos(in_file,'train_g_test_g_5_pos',pos1, pos2)
# For Model 2: greed, backtest in neutral 
pos1 = GP.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 = GP.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
GP.five_label_pos(in_file,'train_g_test_n_5_pos',pos1, pos2)
# For Model 3: fear model, backtest in fear 
pos1 = GP.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 = GP.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
GP.five_label_pos(in_file,'train_f_test_f_5_pos',pos1, pos2)
# For Model 4: fear model, backtest in neutral 
pos1 = GP.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 = GP.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
GP.five_label_pos(in_file,'train_f_test_n_5_pos',pos1, pos2)

##-----------------------------------------------backtesting----------------------------------------## 
price_path = './Data/interval_data.csv'
pos_paths = ['./Prediction/train_g_test_g_5_pos.csv','./Prediction/train_g_test_n_5_pos.csv',
            './Prediction/train_f_test_f_5_pos.csv', './Prediction/train_f_test_n_5_pos.csv']
for pos_path in pos_paths: 
    backtesting(price_path, pos_path)

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
