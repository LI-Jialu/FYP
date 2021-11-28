import datetime as datetime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from download_order_book import download_order_book as dob
import download_price as dp 
from svm import svm_timepoint, svm_interval, train_test
from trading import TestStrategy, CommisionScheme

import backtrader as bt 
import backtrader.feeds as btfeeds
import analysis as analysis
import pyfolio as pf



starttime_0 = datetime.datetime.now()
##-------------------------------------------Data collection-----------------------------------------## 
dob.download_order_book() 
raw_data_path = './raw_data' # folder to store raw data
tickers_list = ['BTC','ETH']
tf = '1m' # 1 minute 
DP = dp.download_price(tickers_list, tf, raw_data_path)
DP.get_raw()
DP.get_file('close')
DP.get_file('volume')
DP.combine_csv()
tt = train_test()
fng = np.array(pd.read_csv('./Data/fng_index.csv', header = 0)['fng_value'])

##---------------------------------Data loading & preprocessing--------------------------------------## 
# pd.set_option('display.max_columns', None)
path = 'C:/tardis_dataset/bitmex'
date_list = ['2021-09-23', '2021-09-24', '2021-09-25', '2021-09-26', '2021-09-27', '2021-09-28', 
             '2021-09-29', '2021-09-30', '2021-10-01', '2021-10-02', '2021-10-03', '2021-10-04', 
             '2021-10-05', '2021-10-06', ]
day_num = len(date_list)
df = [dob.load_data(path, date_list[i]) for i in range(day_num)]

##----------------------------------------SVM Single timepoint-------------------------------------------##
starttime_point = datetime.datetime.now()
svm_s = svm_timepoint(df)
f1, f2, f3, f4, f5, f6 = svm_s.timpoint_feature()
endtime_point = datetime.datetime.now()
print('The SVM using sliding window running time:')
print((endtime_point - starttime_point).seconds)

##----------------------------------------SVM Interval (3 labels)---------------------------------------## 
interval = 5 #10, 20
starttime_interval = datetime.datetime.now()
svm_i = svm_interval(df, interval)
f1, f2, f3, f4, f5, f6, f7 = svm_i.interval_feature()
X = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7)
y = svm_i.generate_y(f2, X.shape[0], 3)
X_train, X_test, y_train, y_test = tt.train_test_split(X, y)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window running time:')
print((endtime_interval - starttime_interval).seconds)

##----------------------------------------SVM Interval (5 labels)--------------------------------------## 


##-------------------------------------------SVM Interval with PCA-------------------------------------## 

##-----------------------------------SVM Interval with PCA with condition ----------------------------## 
interval = 5 #10, 20
# the number of targeted principal componenets
pcs = 5 # 2, 10
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
    svm_i = svm_interval(df[0], interval)
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
svm_model_greed = tt.trian(X_train_greed, y_train_greed)
svm_model_fear = tt.train(X_train_fear, y_train_fear)
endtime_interval = datetime.datetime.now()
print('The SVM using sliding window running time:')
print((endtime_interval - starttime_interval).seconds)


##-----------------------------------------------backtesting------------------------------------------## 
# Initial cash 
cash = 1000000
# Custom commission
comminfo = CommisionScheme(
    commission=0.001,  # 0.1%
    mult=10,
    margin=2000  # Margin is needed for futures-like instruments
)

cerebro = bt.Cerebro()
cerebro.broker.setcash(cash)
cerebro.broker.addcommissioninfo(comminfo)

'''
prediction = np.to_dict(y_pred) 
ticker, target = prediction.items()
datapath = './data'+ticker+'_'+'.csv'
bt_data = btfeeds.GenericCSVData(dataname=datapath,
    fromdate = datetime(2021, 10, 1),
    todate = datetime(2021, 11, 1), 
    nullvalue = 0.0,
    timeframe=bt.TimeFrame.Minutes,
    compression=1,
    dtformat=('%Y-%m-%d %H:%M:%S'),
    datetime=0,
    Open = 1,
    High = 2,
    Low = 3,
    Close = 4,
    Volume = 5, 
    openinterest=-1)
bt_data.target = target
cerebro.adddata(bt_data, name=ticker)
'''

# Add a pyfolio analyzer to view the performance tearsheets
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addstrategy(TestStrategy)

# Run startegy
start_time = datetime.now()
print('Starting Balance: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
ending_value = cerebro.broker.getvalue()
print(f'Final Portfolio Value: {ending_value:,.2f}')
print("--- %s seconds ---" % (datetime.now() - start_time))

# View Pyfolio results
strat = results[0]
pyfoliozer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions,
    live_start_date='2020-06-01',  # This date is sample specific
    round_trips=True)

# backtrader plot
cerebro.plot()
figure = cerebro.plot(style='candlebars')[0][0]
figure.savefig(f'backtrader.png')


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
