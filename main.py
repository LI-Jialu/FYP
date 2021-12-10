import datetime as datetime
import numpy as np 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from download_order_book import download_order_book as dob
from model_builder import model_builder, conditional_model_builder
from generate_pos import generate_pos as gp
#from backtesting import backtest
from trading import CommisionScheme, Strategy
import backtrader as bt
import backtrader.feeds as btfeeds
import analysis as analysis
import pyfolio as pf
from datetime import datetime as dt 

##----------------------------------------------Parameters--------------------------------------------## 
cash = 10000000000
interval = 500 #500, 1000, 2000
pcs = 5 # 2, 10, the number of targeted principal componenets
starttime_0 = dt.now()
##-------------------------------------------Data collection-----------------------------------------## 
DOB = dob()
DOB.download_order_book() 
# Fear and Greedy index 
fng = np.array(pd.read_csv('./Data/fng_index.csv', header = 0)['fng_value'])
pd.set_option('display.max_columns', None)
# path = 'C:/tardis_dataset/bitmex'
path = './Data/Order_book'
print(os.chdir('..'))

##---------------------------------Train Models without condition-------------------------------------##
mb = model_builder(pcs, interval, './Data/Order_book')
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
# pred = pd.DataFrame(gp.load_model(file = 'C:/Users/Jialu/Documents/Code/FYP/metrics/pred_y_list')[0]).to_csv('C:/Users/Jialu/Documents/Code/FYP/metrics/pred_y_list.csv')
pred = gp.load_model(file = 'C:/Users/Jialu/Documents/Code/FYP/metrics/pred_y_list')
threshold1 = 5e-04 
threshold2 = 1e-03
pre_acc_1, pre_acc_m1 = 0.8, 0.8
pre_acc_2, pre_acc_m2 = 0.9, 0.9
# For Model 1: greed, backtest in greed 
pos1 =gp.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 =gp.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)

df = pd.read_csv('C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv',index_col=0)
df['prediction'] = np.random.choice([-1,0,1], size = len(df), p=[0.3,0.4, 0.3])
df.to_csv('C:/Users/Jialu/Documents/Code/FYP/Prediction/train_g_test_g_5.csv')
gp.five_label_pos('train_g_test_g_5','train_g_test_g_5_pos',cash*0.05, cash*0.1)

# For Model 2: greed, backtest in neutral 
# Further trained in GPU version code 
'''
pos1 =gp.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 =gp.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
gp.five_label_pos(in_file,'train_g_test_n_5_pos',pos1, pos2)
# For Model 3: fear model, backtest in fear 
pos1 =gp.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 =gp.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
gp.five_label_pos(in_file,'train_f_test_f_5_pos',pos1, pos2)
# For Model 4: fear model, backtest in neutral 
pos1 =gp.kelly_criteria(pre_acc_1, pre_acc_m1, threshold1, -threshold1)
pos2 =gp.kelly_criteria(pre_acc_2, pre_acc_m2, threshold2, -threshold2)
gp.five_label_pos(in_file,'train_f_test_n_5_pos',pos1, pos2)
'''

##-----------------------------------------------backtesting----------------------------------------## 
'''
pos_paths = ['./Prediction/train_g_test_g_5_pos.csv','./Prediction/train_g_test_n_5_pos.csv',
            './Prediction/train_f_test_f_5_pos.csv', './Prediction/train_f_test_n_5_pos.csv']
for pos_path in pos_paths: 
    backtest.backtesting(price_path, pos_path)
'''

price_path = 'C:/Users/Jialu/Documents/Code/FYP/Data/interval_data.csv'
pos_path = 'C:/Users/Jialu/Documents/Code/FYP/Prediction/train_g_test_g_5_pos.csv'
cerebro = bt.Cerebro( preload=False)
cerebro.broker.setcash(cash)
cerebro.broker.addcommissioninfo(CommisionScheme(commission=0.0004,automargin = 1))
bt_data = btfeeds.GenericCSVData(dataname=price_path,
        fromdate = datetime.datetime(2021, 8, 6),
        todate = datetime.datetime(2021, 8, 7), 
        nullvalue = 0.0,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0,
        openinterest = -1)
target = pd.read_csv(pos_path, usecols=['timestamp','cur_pos'])
target.index = target['timestamp']
target.index = [str(x) for x in target.index]
target = target.to_dict().values()
bt_data.target = target
cerebro.adddata(bt_data)
    

# Add a pyfolio analyzer to view the performance tearsheets
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addstrategy(Strategy)

# Run startegy
start_time = dt.now()
print('Starting Balance: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
ending_value = cerebro.broker.getvalue()
print(f'Final Portfolio Value: {ending_value:,.2f}')
print("--- %s seconds ---" % (dt.now() - start_time))


pyfoliozer = results[0].analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
print(returns)
print(positions)
print(transactions)
print(gross_lev)
# pf.create_full_tear_sheet(returns)

# backtrader plot
cerebro.plot()
figure = cerebro.plot(style='candlebars')[0][0]
figure.savefig(f'./Images/backtrader.png')# View Pyfolio results

##-----------------------------------------------analysis------------------------------------------## 
# prediction report generated by SKlearn 
# testing different time interval selection 
# testing different PCA dimension 
# view Pyfolio results

##-------------------------------------------------End---------------------------------------------## 
endtime_0 = dt.now()
print((endtime_0 - starttime_0).seconds)
print('The whole running time:')
