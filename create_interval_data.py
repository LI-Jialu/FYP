import numpy as np
import pandas as pd
from download_order_book import download_order_book as dob
import os 
import matplotlib.pyplot as plt

def change_date(df): 

    return df
# 500 is too small 
interval = 1000
# dates 
dates = ['2021-07-30', '2021-07-31', '2021-08-01', '2021-08-02', '2021-08-03', '2021-08-04', '2021-08-05', '2021-08-06', '2021-08-07', '2021-08-08', '2021-08-09', '2021-08-10', '2021-08-11', '2021-08-12', '2021-08-13', '2021-08-14', '2021-08-15', '2021-08-16', '2021-08-17', '2021-08-18', '2021-08-19', '2021-08-20', '2021-08-21', '2021-08-22', '2021-08-23', '2021-08-24', '2021-08-25', '2021-08-26', '2021-08-27', '2021-08-28', '2021-08-29', '2021-08-30', '2021-08-31', '2021-09-01', '2021-09-02', '2021-09-03', '2021-09-04', '2021-09-05', '2021-09-06', '2021-09-07', '2021-09-08', '2021-09-09', '2021-09-10', '2021-09-11', '2021-09-12', '2021-09-13', '2021-09-14', '2021-09-15', '2021-09-16', '2021-09-17', '2021-09-18', '2021-09-19', '2021-09-20', '2021-09-21', '2021-09-22', '2021-09-23', '2021-09-24', '2021-09-25', '2021-09-26', '2021-09-27', '2021-09-28', '2021-09-29', '2021-09-30', '2021-10-01', '2021-10-02', '2021-10-03', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07', '2021-10-08', '2021-10-09', '2021-10-10', '2021-10-11', '2021-10-12', '2021-10-13', '2021-10-14', '2021-10-15', '2021-10-16', '2021-10-17', '2021-10-18', '2021-10-19', '2021-10-20', '2021-10-21', '2021-10-22', '2021-10-23', '2021-10-24', '2021-10-25', '2021-10-26', '2021-10-27', '2021-10-28', '2021-10-29', '2021-10-30', '2021-10-31', '2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04', '2021-11-05', '2021-11-06', '2021-11-07', '2021-11-08', '2021-11-09', '2021-11-10', '2021-11-11', '2021-11-12', '2021-11-13', '2021-11-14', '2021-11-15', '2021-11-16', '2021-11-17', '2021-11-18', '2021-11-19', '2021-11-20', '2021-11-21', '2021-11-22', '2021-11-23', '2021-11-24', '2021-11-25', '2021-11-26', '2021-11-27', '2021-11-28', '2021-11-29']
print(dates[7]) #2021-08-06 -> 2021-
print(dates[62]) #2021-09-30 -> 2021-11-01
print(dates[95]) #2021-11-02
# load data 1 
DOB = dob()
path = 'C:/Users/Jialu/Documents/Code/FYP/Data/Order_book'
df_1 = DOB.load_data(path, dates[7]).iloc[:,:10]
rem = df_1.shape[0] % interval
window_num = (df_1.shape[0]-rem)/interval 
df_1 = df_1.drop(df_1.tail(rem).index)
'''
# load data 2 
df_2 = DOB.load_data(path, dates[62]).iloc[:,:10]
rem = df_2.shape[0] % interval
window_num = (df_2.shape[0]-rem)/interval 
df_2 = df_2.drop(df_2.tail(rem).index)

# load data 3 
df_3 = DOB.load_data(path, dates[95]).iloc[:,:10]
rem = df_3.shape[0] % interval
window_num = (df_2.shape[0]-rem)/interval 
df_3 = df_3.drop(df_3.tail(rem).index)

# combine all data and then split 
pdList = [df_1, df_2, df_3] 
df = pd.concat(pdList)
'''
df = df_1
# split dataframe into intervals 
splited = np.array_split(df,window_num)
splited_array = [pd.DataFrame( s.iloc[-1,:]) for s in splited]
splited_df = pd.DataFrame()
for s in splited_array:
    splited_df = pd.concat([splited_df, s],axis = 1)
splited_df = splited_df.T
# 0: timestamp,1: Pa1,5:Pa2,7:Pb2,3:Pb1, 2:Va1
#             ('open', 1), ('high', 2), ('low', 3), ('close', 4),
to_csv_data = splited_df.iloc[:,[0,1,5,7,3,2]]
to_csv_data.to_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv')
df = pd.read_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv',usecols=['timestamp','Pa1','Pa2','Pb2','Pb1','Va1'],index_col='timestamp')
df.index = pd.to_datetime(df.index)
df.index = [x.replace(microsecond=0) for x in df.index]
df.to_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv')