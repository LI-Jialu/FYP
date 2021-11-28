# pip install tardis-client
import asyncio
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
  async for local_timestamp, message in messages:
    print(message)


asyncio.run(replay())