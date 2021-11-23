import wget
import zipfile
import pandas as pd
from datetime import datetime
import os
import glob


class get_data:
    # Input: Ticker List = Name of the crypto currency eg. ETH / 1000SHIB
    # tf = Time Frame eg. 15m , 4h, 1d
    # path = path for temporily saving the raw OHLCV data (Please make sure that there is nothing inside this folder!!)

    def __init__(self, ticker_list, tf, path):
        self.ticker_list = ticker_list
        self.tf = tf
        self.path = path
        self.col_name = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av',
                         'tb_quote_av', 'ignore']

    def get_raw(self):

        year = [2021]
        month = [str(1).zfill(2), str(2).zfill(2), str(3).zfill(2), str(4).zfill(2), str(5).zfill(2), str(6).zfill(2),
                 str(7).zfill(2), str(8).zfill(2), str(9).zfill(2), 10, 11, 12]

        for ticker in self.ticker_list:
            folder_path = self.path + '/' + ticker + '/' + self.tf
            if not os.path.exists(folder_path):
                os.makedirs(folder_path) # Creating the folder
            for y in year:
                for m in month:
                    filename = ticker + 'USDT' + '-' + self.tf + '-' + str(y) + '-' + str(m) + '.zip'

                    try:
                        url = 'https://data.binance.vision/data/futures/um/monthly/klines/' + ticker + 'USDT/' + self.tf + '/' + filename
                        wget.download(url, folder_path)
                        with zipfile.ZipFile(folder_path + '/' + filename, 'r') as zip_ref:
                            zip_ref.extractall(folder_path)
                        os.remove(folder_path + '/' + filename)
                    except:
                        None
        return

    def get_file(self, col):
        df_all = pd.DataFrame()

        for ticker in self.ticker_list:
            folder_path = self.path + '/' + ticker + '/' + self.tf
            dir_list = os.listdir(folder_path)
            df = pd.DataFrame()
            for file in dir_list:
                data = pd.read_csv(folder_path + '/' + file, header=None, names=self.col_name)
                df = pd.concat([df, data[col]], axis=0)  # get close/volume price

            time = [datetime.fromtimestamp(int(t)) for t in df.index / 1000]
            df.index = time
            df.sort_index()
            df_all = pd.concat([df_all, df], axis=1)

        df_all.columns = self.ticker_list
        df_all.to_csv('crypto_' + self.tf + '_' + col + '.csv')
        return

    def load_data(path, date):
        df = pd.read_csv(path + '/book_snapshot_25/XBTUSD/bitmex_book_snapshot_25_' + date + '_XBTUSD.csv.gz',
                        header = 0,
                        names = ['timestamp', 'Pa1', 'Va1', 'Pb1', 'Vb1', 'Pa2', 'Va2', 'Pb2', 'Vb2', 
                                'Pa3', 'Va3', 'Pb3', 'Vb3', 'Pa4', 'Va4', 'Pb4', 'Vb4', 
                                'Pa5', 'Va5', 'Pb5', 'Vb5', 'Pa6', 'Va6', 'Pb6', 'Vb6', 
                                'Pa7', 'Va7', 'Pb7', 'Vb7', 'Pa8', 'Va8', 'Pb8', 'Vb8', 
                                'Pa9', 'Va9', 'Pb9', 'Vb9', 'Pa10', 'Va10', 'Pb10', 'Vb10'],
                        usecols = [2] + list(range(4, 44)),
                        compression = 'gzip')
        return df

