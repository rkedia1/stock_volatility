'''
@author Evan McFall
'''

import pandas as pd
# from date.relativedelta import realativedelta
import requests
import csv
import os
from tqdm import tqdm
import sqlite3

from yfinance import download as ydownload
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')


class Equities(object):
    _equities = None

    stock_filename = 'equities/all_stocks.csv'
    sector_filename = 'equities/sector.csv'

    # TODO secrets if for use in future
    finnhub_api_key = 'chrb7qhr01qkb63asu2gchrb7qhr01qkb63asu30'

    # temporary use case
    _consumer_stocks = None

    def __init__(self):
        pass
        # tickers_list = ['AAPL', 'IBM', 'MSFT', 'GOOG']
        #
        # for ticker in tickers_list:
        #     stock = yf.Ticker(ticker)
        #     print(stock.info)
        #     industry = info.get('industry', '')
        #     print(industry)

    def get_all_us_equities(self):
        # relative import to avoid any issues with packages not needed
        # given that the list has been populated and published
        import finnhub

        finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
        stock_data = finnhub_client.stock_symbols(exchange='US')
        all_stocks = [x['displaySymbol'] for x in stock_data
                      if x['type'] == 'Common Stock' and x['currency'] == 'USD'
                      and x['mic'] in ['XNAS', 'XNYS']]
        return all_stocks

    @property
    def all_stocks(self):
        '''
        Should in theory only be run once if the stocks have not been created as a CSV
        Pulls in a full list of equities then appends the associated data from yahoo finance
        :return:
        '''
        if not os.path.isfile(self.stock_filename):
            all_stocks = self.get_all_us_equities()

            dataset = pd.DataFrame()
            for ticker in tqdm(all_stocks):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.info
                    data['Symbol'] = ticker
                    # .append to be depricated; but still going to use it
                    dataset = dataset.append(pd.DataFrame([data], index=[ticker]))

                    # constantly writing as it is not exhaustive and in the cause of failure
                    dataset.to_csv('equities/all_stocks.csv')
                except: pass


class EquityData(Equities):
    db_name = 'database.db'

    def __init__(self):
        Equities.__init__(self)

    @staticmethod
    def batch(iterable, n=1):
        for ndx in range(0, len(iterable), n):
            yield iterable[ndx:min(ndx + n, len(iterable))]

    def download_ticker_data(self, stocks: list,
                             interval: str = '1h',
                             period: str = '5y'):
        '''
        yf.download(tickers = "SPY AAPL",  # list of tickers
            period = "1y",         # time period
            interval = "1d",       # trading interval
            prepost = False,       # download pre/post market hours data?
            repair = True)         # repair obvious price errors e.g. 100x?
        '''

        # this sets up the longest potential query as intraday only goes back 2 years
        # may present an issue with number of samples
        stocks = list(stocks)
        if not any(x in interval.lower() for x in ['y', 'm', 'd']):
            period = '2y'

        for batch_stocks in self.batch(stocks, 100):
            subset = dict()
            data = ydownload(batch_stocks, period=period, interval=interval, group_by='ticker')
            for stock in batch_stocks:
                try:
                    subset[stock] = data[stock]
                except: pass

            for key, df in subset.items():
                with sqlite3.connect(self.db_name) as conn:
                    table_name = f'{key}_{interval.lower()}'
                    df.to_sql(table_name, conn, if_exists='replace')

    def stock_data(self, interval: str = '1h') -> pd.DataFrame:
        '''
        This will provide the actual data needed for a frequency
        :param interval:
        :return:
        '''
        dataset = dict()
        for stock in self.consumer_stocks:
            try:
                table_name = f'{stock}_{interval.lower()}'
                query = f'SELECT * FROM {table_name}'
                with sqlite3.connect(self.db_name) as conn:
                    df = pd.read_sql(query, con=conn)
                    dataset[stock] = df
            except Exception as e:
                print(e)

    @property
    def consumer_stocks(self):
        # hardcoded
        if self._consumer_stocks is None:
            disc = pd.read_csv('equities/discretionary.csv')
            staples = pd.read_csv('equities/staples.csv')
            # again I will use .append to the end
            stocks = disc.append(staples)['Symbol'].to_numpy()
            self._consumer_stocks = stocks

        return self._consumer_stocks

    def update_data(self) -> pd.DataFrame:
        '''
        Explicit for this use case
        :return: pd.DataFrame
        '''

        stocks = self.consumer_stocks
        self.download_ticker_data(stocks, '1h')
        self.download_ticker_data(stocks, '1d')


if __name__ == '__main__':
    EquityData().update_data()