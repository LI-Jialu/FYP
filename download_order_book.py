import os 
from tardis_dev import datasets, get_exchange_details
import logging
import pandas as pd
from datetime import datetime as dt


class download_order_book: 
    def __init__(self):
        pass

    def default_file_name(exchange, data_type, date, symbol, format):
        return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"

    def file_name_nested(exchange, data_type, date, symbol, format):
        return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"

    def download_order_book(self): 
        os.chdir('.\Data')
        logging.basicConfig(level=logging.DEBUG)
        datasets.download(
            exchange="binance-futures",
            # 'dataType' param values: 
            # 'trades', 'incremental_book_L2', 'quotes', 'derivative_ticker', 'options_chain', 'book_snapshot_5', 'book_snapshot_25', 'liquidations'.
            data_types=['book_snapshot_5'],
            # filters=[Channel(name="depth", symbols=["btcusdt"])],
            from_date="2021-07-30",
            # to date is non inclusive
            to_date="2021-11-30",
            symbols=["BTCUSDT"],
            api_key="TD.P1RQy3kV6rkCX-Js.8wpD7WjlIBu6l5O.wujparLjM0uHh5P.haoaBrqe-I8WegV.9mLshEQh6GT1INZ.8qOm",
            download_dir="./Order_book",
            # get_filename=default_file_name,
            # get_filename=file_name_nested,
            
        )

    '''def download_derivative_ticker(self): 
        os.chdir('.\Data')
        logging.basicConfig(level=logging.DEBUG)
        datasets.download(
            exchange="binance-futures",
            # Allowed 'dataType' param values: 
            # 'trades', 'incremental_book_L2', 'quotes', 'derivative_ticker', 'options_chain', 'book_snapshot_5', 'book_snapshot_25', 'liquidations'.
            data_types=['derivative_ticker'],
            from_date="2021-10-02",
            # to date is non inclusive
            to_date="2021-10-03",
            symbols=["BTCUSDT"],
            api_key="TD.qtKSUEXoqaY7HYJC.WbIkzzx6IlUzmfW.HpGRMPQvrzWmja0.ufinV2kPJLc8WTl.1Nzl5-0NRFZkP7m.3BdA",
            download_dir="./Derivative_ticker",            
        )'''

    # binance-futures_book_snapshot_5_2021-09-30_BTCUSDT.csv
    def load_data(self, path, date):
        df = pd.read_csv(path + '/binance-futures_book_snapshot_5_' + date + '_BTCUSDT.csv.gz',
                        header = 0,
                        names = ['timestamp', 'Pa1', 'Va1', 'Pb1', 'Vb1', 'Pa2', 'Va2', 'Pb2', 'Vb2', 
                                'Pa3', 'Va3', 'Pb3', 'Vb3', 'Pa4', 'Va4', 'Pb4', 'Vb4', 
                                'Pa5', 'Va5', 'Pb5', 'Vb5'],
                        usecols = [2] + list(range(4, 24)),
                        compression = 'gzip' 
                        )
        # print(df['timestamp'])
        df['timestamp'] = [str(x)[:-6]+'.'+str(x)[-6:] for x in df['timestamp']]
        # print(df['timestamp'][0])
        df['timestamp'] = [dt.fromtimestamp(float(x)) for x in df['timestamp']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

