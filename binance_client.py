import os

from binance.client import Client
from time import sleep

from binance import ThreadedWebsocketManager
import pandas as pd

# init
api_key = 'vzHiMIbYf3JvTj89t7Q2fzXINB9F7uSDVGHOvw6KeuZx2pV9W27Sj3iW0GPClaVJ'
api_secret = 'fspuQPZSzS002ljbt75t1AfhAdHXv8o0Ry2sWuDxaUoGilwwuuOpw0SbQcWvQ5PG'

client = Client(api_key, api_secret)

client.API_URL = 'https://testnet.binance.vision/api'
# client.API_URL = 'https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000'

# get latest price from Binance API
btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
# print full output (dictionary)
print(btc_price)


# get Bitcoin’s historical price data in CSV format
# valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# get timestamp of earliest date data is available
timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1d')
print(timestamp)
# request historical candle (or klines) data
bars = client.get_historical_klines('BTCUSDT', '1d', timestamp, limit=1000)

# option 4 - create a Pandas DataFrame and export to CSV
btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
btc_df.set_index('date', inplace=True)
print(btc_df.head())
# export DataFrame to csv
btc_df.to_csv('btc_bars.csv')