import os 
from tardis_dev import datasets, get_exchange_details
import logging


class data_collection: 
    def __init__(self) -> None:
        pass

    # function used by default if not provided via options
    def default_file_name(exchange, data_type, date, symbol, format):
        return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


    # customized get filename function - saves data in nested directory structure
    def file_name_nested(exchange, data_type, date, symbol, format):
        return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"

    def download(self): 
        # os.chdir('.\Data')
        # needed data: 
        # snapshot_25, derivative_ticker 
        logging.basicConfig(level=logging.DEBUG)
        datasets.download(
            # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
            exchange="bitmex",
            # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
            # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
            # data_types=["incremental_book_L2", "trades", "quotes", "derivative_ticker", "book_snapshot_25", "liquidations"],
            data_types=[ "book_snapshot_25","incremental_book_L2","derivative_ticker"],
            # change date ranges as needed to fetch full month or year for example
            from_date="2021-09-23",
            # to date is non inclusive
            to_date="2021-10-07",
            # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
            symbols=["XBTUSD"],
            # (optional) your API key to get access to non sample data as well
            api_key="TD.qtKSUEXoqaY7HYJC.WbIkzzx6IlUzmfW.HpGRMPQvrzWmja0.ufinV2kPJLc8WTl.1Nzl5-0NRFZkP7m.3BdA",
            # (optional) path where data will be downloaded into, default dir is './datasets'
            download_dir="./",
            # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
            # get_filename=default_file_name,
            # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
            # get_filename=file_name_nested,
        )