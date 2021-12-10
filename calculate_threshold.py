import numpy as np
import pandas as pd
from download_order_book import download_order_book as dob
import os 
import matplotlib.pyplot as plt

interval = 1000 #500， 1000， 2000
path = 'C:/Users/Jialu/Documents/Code/FYP/'

##---------------------------------------------load Data------------------------------------------------------------------## 
DOB = dob()
df = DOB.load_data(path + 'Data/Order_book', '2021-09-30').iloc[:,:10]
rem = df.shape[0] % interval
window_num = (df.shape[0]-rem)/interval 
df = df.drop(df.tail(rem).index)

# split dataframe into intervals and select data for backtesting 
splited = np.array_split(df,window_num)
splited_array = [pd.DataFrame( s.iloc[-1,:]) for s in splited]
splited_df = pd.DataFrame()
for s in splited_array:
    splited_df = pd.concat([splited_df, s],axis = 1)
splited_df.T.iloc[:,[0,1,3,5,7,2]].to_csv(path + 'Data/interval_data.csv')
df = pd.read_csv(path + 'Data/interval_data.csv',usecols=['timestamp','Pa1','Pb1','Pa2','Pb2','Va1'],index_col='timestamp')
df.index = pd.to_datetime(df.index)
df.index = [x.replace(microsecond=0) for x in df.index]
df.to_csv(path + 'Data/interval_data.csv')
# calculate each interval's pct change 
ret = splited_df.iloc[:,1:].pct_change().dropna()

##---------------------------------percentage change distribution --------------------------------------------------------##
print(ret)
x1 = ret.iloc[:,0]
x2 = ret.iloc[:,1]
x3 = ret.iloc[:,2]
x4 = ret.iloc[:,3]

## distribution graph 
kwags = dict(alpha=0.3) 
plt.hist(x1, color='red',label='ask', **kwags)
plt.hist(x3, color = 'blue',label ='bid', **kwags)
plt.xlabel('Percentage change',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Price change of asks and bids in an interval',fontsize=14)
plt.legend(fontsize=10)
fig = plt.gcf()
fig.set_size_inches(7.2, 4.2)
fig.savefig('./Price_change.png', dpi=100)


'''plt.hist(x2, color='green',label='ask', **kwags)
plt.hist(x4, color = 'yellow',label ='bid', **kwags)
plt.xlabel('Percentage change',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Volumn change of asks and bids in an interval',fontsize=14)
plt.legend(fontsize=10)
fig = plt.gcf()
fig.set_size_inches(7.2, 4.2)
fig.savefig('./Volume_change.png', dpi=100)'''


counts, bin_edges = np.histogram(x1, bins=20)
print(counts, bin_edges)

counts, bin_edges = np.histogram(x2, bins=5)
print(counts, bin_edges)

counts, bin_edges = np.histogram(x3, bins=20)
print(counts, bin_edges)

counts, bin_edges = np.histogram(x4, bins=5)
print(counts, bin_edges)