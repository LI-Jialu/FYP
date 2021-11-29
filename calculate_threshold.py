import numpy as np
import pandas as pd
from download_order_book import download_order_book as dob


interval = 500

# load data 
DOB = dob()
path = './Data/Order_book'
df = DOB.load_data(path, '2021-10-02')
rem = df.shape[0] % interval
window_num = (df.shape[0]-rem)/interval 
df = df.drop(df.tail(rem).index)

# split dataframe into intervals 
np.array_split(df, 3)


# each interval's pct change 

# pct change distribution 

# pct change count or statistical calculation 