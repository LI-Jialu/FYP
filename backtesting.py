import backtrader as bt 
import backtrader.feeds as btfeeds
import analysis as analysis
import pyfolio as pf
from trading import TestStrategy, CommisionScheme
import datetime as datetime
import numpy as np 
import pandas as pd

class backTest():

    

    # read predction csv 
    y_pred = pd.readcsv('./Data/predictions')
    '''y_pred_dict = np.to_dict(y_pred) 
    pred = y_pred_dict.items()'''

    # read price csv 
    datapath = './Data/Price/price.csv'

    # from prediction to trading position 

    # Initialize cerebro 
    cerebro = bt.Cerebro()
    cash = 1000000
    cerebro.broker.setcash(cash)
    commSheme = CommisionScheme(commission=0.0004,automargin = 1)
    cerebro.broker.addcommissioninfo(commSheme)

    bt_data = btfeeds.GenericCSVData(dataname=datapath,
        fromdate = datetime(2021, 9, 23),
        todate = datetime(2021, 10, 7), 
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

