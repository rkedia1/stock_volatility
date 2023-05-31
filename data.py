import pandas as pd
# from date.relativedelta import realativedelta
import requests
import csv
import os
from tqdm import tqdm

from yfinance import download
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')


class Equities(object):
    _equities = None

    stock_filename = 'equities/all_stocks.csv'
    sector_filename = 'equities/sector.csv'

    def __init__(self):
        pass
        # tickers_list = ['AAPL', 'IBM', 'MSFT', 'GOOG']
        #
        # for ticker in tickers_list:
        #     stock = yf.Ticker(ticker)
        #     print(stock.info)
        #     industry = info.get('industry', '')
        #     print(industry)

    @property
    def all_stocks(self):
        if not os.path.isfile(self.stock_filename):
            import finnhub
            api_key = 'chrb7qhr01qkb63asu2gchrb7qhr01qkb63asu30'
            finnhub_client = finnhub.Client(api_key=api_key)
            stock_data = finnhub_client.stock_symbols(exchange='US')
            all_stocks = [x['displaySymbol'] for x in stock_data
                          if x['type'] == 'Common Stock' and x['currency'] == 'USD'
                          and x['mic'] in ['XNAS', 'XNYS']]

            dataset = pd.DataFrame()
            for ticker in tqdm(all_stocks):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.info
                    data['Symbol'] = ticker
                    dataset = dataset.append(pd.DataFrame([data], index=[ticker]))
                    dataset.to_csv('equities/all_stocks.csv')
                except: pass


class EquityData(Equities):
    def __init__(self):
        Equities.__init__(self)

    def download(self):
        pass




if __name__ == '__main__':
    print(Equities().all_stocks)