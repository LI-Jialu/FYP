# pip install tardis-client
import asyncio
import csv 
from tardis_client import TardisClient, Channel
tardis_client = TardisClient(api_key="TD.qtKSUEXoqaY7HYJC.WbIkzzx6IlUzmfW.HpGRMPQvrzWmja0.ufinV2kPJLc8WTl.1Nzl5-0NRFZkP7m.3BdA")

async def replay():
  # replay method returns Async Generator
  messages = tardis_client.replay(
    exchange="binance-futures",
    from_date="2021-10-02",
    to_date="2021-10-03",
    filters=[Channel(name="markPrice", symbols=["btcusdt"])]
  )

  # messages as provided by Binance USDT Futures real-time stream
  with open("./Data/csv_file.csv", "w") as f: 
    writer = csv.writer(f)
    writer.writerow(['timestamp','p'])
    async for local_timestamp, message in messages:
      writer.writerow([local_timestamp, message.get('data').get('p'),  message.get('data').get('P')])
    


asyncio.run(replay())