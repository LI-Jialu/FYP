import datetime as datetime
import numpy as np 
import os 
import pandas as pd
from download_order_book import download_order_book as dob
from model_builder_2 import model_builder_2 as model_builder
from model_builder_2 import conditional_model_builder_2 as conditional_model_builder
import pickle

def my_dump(obj, fname): 
    try:
        fp=open('./metrics/' + fname,'wb')
        pickle.dump(obj,fp)
        fp.close()
    except Exception as e:
        print(e)

starttime_0 = datetime.datetime.now()
interval = 1000 #500, 1000, 2000
pcs = 5 # 2, 10, the number of targeted principal componenets

DOB = dob()
# DOB.download_derivative_ticker() 
DOB.download_order_book()
print('successfully download!')

path = './Data/Order_book'

##-----------------------------------SVM Interval with PCA with condition ----------------------------## 
print('Job 2 Begins ')
cmb = conditional_model_builder(pcs, interval, path)
tt_result = cmb.build_model()
pred_y_list = tt_result[:4]
conditional_score_list = tt_result[4:8]
conditional_report_list = tt_result[8:]

##---------------------------------Train Models without condition-------------------------------------##
#print('Job 1 Begins ')
#mb = model_builder(pcs, interval, path)
#tt_result = mb.build_model()
#score_list = tt_result[:4]
#report_list = tt_result[4:]

#my_dump(score_list, 'score_list')
#my_dump(report_list, 'report_list')
my_dump(pred_y_list, 'pred_y_list')
my_dump(conditional_score_list, 'c_score_list')
my_dump(conditional_report_list, 'c_report_list')


endtime_0 = datetime.datetime.now()
print((endtime_0 - starttime_0).seconds)
print('The whole running time:')