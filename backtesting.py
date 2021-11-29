from trading import CommisionScheme, TestStrategy
import backtrader as bt
import backtrader.feeds as btfeeds
import analysis as analysis
import pyfolio as pf
from datetime import datetime

class backtest():
    def __init__(self) -> None:
        pass

    def backtesting():
        # Initialize cerebro 
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(1000000)
        cerebro.broker.addcommissioninfo(CommisionScheme(commission=0.0004,automargin = 1))
        bt_data = btfeeds.GenericCSVData(dataname=datapath,
                fromdate = datetime(2021, 9, 23),
                todate = datetime(2021, 10, 7), 
                nullvalue = 0.0,
                timeframe=bt.TimeFrame.Minutes,
                compression=1,
                dtformat=('%Y-%m-%d %H:%M:%S'),
                datetime=0,
                Open = 1,
                Close = 2,
                openinterest=-1)
        bt_data.target = target
        cerebro.adddata(bt_data)
            

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


        pyfoliozer = results[0].analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        pf.create_full_tear_sheet(
            returns,
            positions=positions,
            transactions=transactions,
            live_start_date='2021-10-02',  # This date is sample specific
            round_trips=True)

        # backtrader plot
        cerebro.plot()
        figure = cerebro.plot(style='candlebars')[0][0]
        figure.savefig(f'backtrader.png')# View Pyfolio results