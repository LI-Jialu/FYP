import wget
import zipfile
import pandas as pd
from datetime import datetime
import os


class download_price:
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
        month = [str(9).zfill(2), 10]

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

    

