'''import pandas as pd
import json
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
resu = urllib.request.urlopen('https://api.alternative.me/fng/?limit=130&format=csv&date_format=us', timeout = 150)
data = resu.read()
# print(data)
data ='fng_value,fng_classification,date\n11-29-2021,33,Fear\n11-28-2021,27,Fear\n11-27-2021,21,Extreme Fear\n11-26-2021,47,Neutral\n11-25-2021,32,Fear\n11-24-2021,42,Fear\n11-23-2021,33,Fear\n11-22-2021,50,Neutral\n11-21-2021,49,Neutral\n11-20-2021,43,Fear\n11-19-2021,34,Fear\n11-18-2021,54,Neutral\n11-17-2021,52,Neutral\n11-16-2021,71,Greed\n11-15-2021,72,Greed\n11-14-2021,74,Greed\n11-13-2021,72,Greed\n11-12-2021,74,Greed\n11-11-2021,77,Extreme Greed\n11-10-2021,75,Greed\n11-09-2021,84,Extreme Greed\n11-08-2021,75,Greed\n11-07-2021,73,Greed\n11-06-2021,71,Greed\n11-05-2021,73,Greed\n11-04-2021,73,Greed\n11-03-2021,76,Extreme Greed\n11-02-2021,73,Greed\n11-01-2021,74,Greed\n10-31-2021,74,Greed\n10-30-2021,73,Greed\n10-29-2021,70,Greed\n10-28-2021,66,Greed\n10-27-2021,73,Greed\n10-26-2021,76,Extreme Greed\n10-25-2021,72,Greed\n10-24-2021,73,Greed\n10-23-2021,74,Greed\n10-22-2021,75,Greed\n10-21-2021,84,Extreme Greed\n10-20-2021,82,Extreme Greed\n10-19-2021,75,Greed\n10-18-2021,78,Extreme Greed\n10-17-2021,79,Extreme Greed\n10-16-2021,78,Extreme Greed\n10-15-2021,71,Greed\n10-14-2021,70,Greed\n10-13-2021,70,Greed\n10-12-2021,78,Extreme Greed\n10-11-2021,71,Greed\n10-10-2021,71,Greed\n10-09-2021,72,Greed\n10-08-2021,74,Greed\n10-07-2021,76,Extreme Greed\n10-06-2021,68,Greed\n10-05-2021,59,Greed\n10-04-2021,54,Neutral\n10-03-2021,49,Neutral\n10-02-2021,54,Neutral\n10-01-2021,27,Fear\n09-30-2021,20,Extreme Fear\n09-29-2021,24,Extreme Fear\n09-28-2021,25,Extreme Fear\n09-27-2021,26,Fear\n09-26-2021,27,Fear\n09-25-2021,28,Fear\n09-24-2021,33,Fear\n09-23-2021,27,Fear\n09-22-2021,21,Extreme Fear\n09-21-2021,27,Fear\n09-20-2021,50,Neutral\n09-19-2021,53,Neutral\n09-18-2021,50,Neutral\n09-17-2021,48,Neutral\n09-16-2021,53,Neutral\n09-15-2021,49,Neutral\n09-14-2021,30,Fear\n09-13-2021,44,Fear\n09-12-2021,32,Fear\n09-11-2021,31,Fear\n09-10-2021,46,Fear\n09-09-2021,45,Fear\n09-08-2021,47,Neutral\n09-07-2021,79,Extreme Greed\n09-06-2021,79,Extreme Greed\n09-05-2021,73,Greed\n09-04-2021,72,Greed\n09-03-2021,74,Greed\n09-02-2021,74,Greed\n09-01-2021,71,Greed\n08-31-2021,73,Greed\n08-30-2021,73,Greed\n08-29-2021,72,Greed\n08-28-2021,78,Extreme Greed\n08-27-2021,71,Greed\n08-26-2021,75,Greed\n08-25-2021,73,Greed\n08-24-2021,79,Extreme Greed\n08-23-2021,79,Extreme Greed\n08-22-2021,76,Extreme Greed\n08-21-2021,78,Extreme Greed\n08-20-2021,70,Greed\n08-19-2021,70,Greed\n08-18-2021,73,Greed\n08-17-2021,72,Greed\n08-16-2021,72,Greed\n08-15-2021,71,Greed\n08-14-2021,76,Extreme Greed\n08-13-2021,70,Greed\n08-12-2021,70,Greed\n08-11-2021,70,Greed\n08-10-2021,71,Greed\n08-09-2021,65,Greed\n08-08-2021,74,Greed\n08-07-2021,69,Greed\n08-06-2021,52,Neutral\n08-05-2021,50,Neutral\n08-04-2021,42,Fear\n08-03-2021,48,Neutral\n08-02-2021,48,Neutral\n08-01-2021,60,Greed\n07-31-2021,60,Greed\n07-30-2021,53,Neutral\n'

fng_list = str(data).split('\n')[1:]
fng_value = [x.split(',',1)[-1].split(',')[0] for x in fng_list][:-1]
# test = json.loads(data.decode('utf-8'))
print(fng_list)
print(fng_value)
print(len(fng_value))'''

import datetime
import pandas as pd
date_list = pd.date_range(start="2021-07-30",end="2021-11-29")
dates = [str(x).replace(' 00:00:00','') for x in date_list]
print(dates)