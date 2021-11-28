import os 
from tardis_dev import datasets, get_exchange_details
import logging
import pandas as pd
from datetime import datetime as dt


class download_order_book: 
    def __init__(self):
        pass

    # function used by default if not provided via options
    def default_file_name(exchange, data_type, date, symbol, format):
        return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


    # customized get filename function - saves data in nested directory structure
    def file_name_nested(exchange, data_type, date, symbol, format):
        return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"

    def download_order_book(self): 
        os.chdir('.\Data')
        logging.basicConfig(level=logging.DEBUG)
        datasets.download(
            # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
            exchange="binance-futures",
            # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
            # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
            # Allowed  'dataType' param values: 
            # 'trades', 'incremental_book_L2', 'quotes', 'derivative_ticker', 'options_chain', 'book_snapshot_5', 'book_snapshot_25', 'liquidations'.
            data_types=['book_snapshot_5'],
            # filters=[Channel(name="depth", symbols=["btcusdt"])],
            from_date="2021-09-23",
            # to date is non inclusive
            to_date="2021-10-07",
            # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
            symbols=["BTCUSDT"],
            # (optional) your API key to get access to non sample data as well
            api_key="TD.qtKSUEXoqaY7HYJC.WbIkzzx6IlUzmfW.HpGRMPQvrzWmja0.ufinV2kPJLc8WTl.1Nzl5-0NRFZkP7m.3BdA",
            # (optional) path where data will be downloaded into, default dir is './datasets'
            download_dir="./Order_book",
            # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
            # get_filename=default_file_name,
            # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
            # get_filename=file_name_nested,
            
        )

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
        print(df['timestamp'])
        df['timestamp'] = [str(x)[:-6]+'.'+str(x)[-6:] for x in df['timestamp']]
        print(df['timestamp'][0])
        df['timestamp'] = [dt.fromtimestamp(float(x)) for x in df['timestamp']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

