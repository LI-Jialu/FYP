import numpy as np
import pandas as pd
from download_order_book import download_order_book as dob
import os 
import matplotlib.pyplot as plt

# 500 is too small 
interval = 1000

# load data 
DOB = dob()
print(os.getcwd())
path = 'C:/Users/Jialu/Documents/Code/FYP/Data/Order_book'
df = DOB.load_data(path, '2021-09-30').iloc[:,:10]
rem = df.shape[0] % interval
window_num = (df.shape[0]-rem)/interval 
df = df.drop(df.tail(rem).index)

# split dataframe into intervals 
splited = np.array_split(df,window_num)
splited_array = [pd.DataFrame( s.iloc[-1,:]) for s in splited]
splited_df = pd.DataFrame()
for s in splited_array:
    splited_df = pd.concat([splited_df, s],axis = 1)
splited_df = splited_df.T
to_csv_data = splited_df.iloc[:,[0,1,3,5,7,2]]
to_csv_data.to_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv')
df = pd.read_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv',usecols=['timestamp','Pa1','Pb1','Pa2','Pb2','Va1'],index_col='timestamp')
df.index = pd.to_datetime(df.index)
df.index = [x.replace(microsecond=0) for x in df.index]

#pd['timestamp'] = [x.replace(microsecond=0) for x in pd['timestamp']]
df.to_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv')
# each interval's pct change 
ret = splited_df.iloc[:,1:].pct_change().dropna()

# pct change distribution 
print(ret)
x1 = ret.iloc[:,0]
x2 = ret.iloc[:,1]
x3 = ret.iloc[:,2]
x4 = ret.iloc[:,3]
'''
kwags = dict(alpha=0.3) 
plt.hist(x1, color='red',label='ask', **kwags)
plt.hist(x3, color = 'blue',label ='bid', **kwags)
plt.xlabel('Percentage change',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Price change of asks and bids in an interval',fontsize=14)
plt.legend(fontsize=10)
fig = plt.gcf()
fig.set_size_inches(7.2, 4.2)
fig.savefig('./Price_change.png', dpi=100)'''


'''plt.hist(x2, color='green',label='ask', **kwags)
plt.hist(x4, color = 'yellow',label ='bid', **kwags)
plt.xlabel('Percentage change',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Volumn change of asks and bids in an interval',fontsize=14)
plt.legend(fontsize=10)
fig = plt.gcf()
fig.set_size_inches(7.2, 4.2)
fig.savefig('./Volume_change.png', dpi=100)'''

# pct change count or statistical calculation 
counts, bin_edges = np.histogram(x1, bins=20)
print(counts, bin_edges)
'''
counts, bin_edges = np.histogram(x2, bins=5)
print(counts, bin_edges)
'''
counts, bin_edges = np.histogram(x3, bins=20)
print(counts, bin_edges)
print('Done')
'''counts, bin_edges = np.histogram(x4, bins=5)
print(counts, bin_edges)'''



