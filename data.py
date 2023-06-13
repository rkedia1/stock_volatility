"""
@author Evan McFall
"""

import pandas as pd
import numpy as np

# from date.relativedelta import realativedelta
import requests
import os
from tqdm import tqdm
import sqlite3
from snscrape.modules.twitter import TwitterSearchScraper, TwitterSearchScraperMode
from datetime import date, timedelta
import csv
from yfinance import download as ydownload
import yfinance as yf
import logging

import warnings

warnings.filterwarnings("ignore")


class Equities(object):
    _equities = None

    stock_filename = "equities/all_stocks.csv"
    sector_filename = "equities/sector.csv"

    # TODO secrets if for use in future
    finnhub_api_key = "chrb7qhr01qkb63asu2gchrb7qhr01qkb63asu30"

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
        stock_data = finnhub_client.stock_symbols(exchange="US")
        all_stocks = [
            x["displaySymbol"]
            for x in stock_data
            if x["type"] == "Common Stock"
            and x["currency"] == "USD"
            and x["mic"] in ["XNAS", "XNYS"]
        ]
        return all_stocks

    @property
    def all_stocks(self):
        """
        Should in theory only be run once if the stocks have not been created as a CSV
        Pulls in a full list of equities then appends the associated data from yahoo finance
        :return:
        """
        if not os.path.isfile(self.stock_filename):
            all_stocks = self.get_all_us_equities()

            dataset = pd.DataFrame()
            for ticker in tqdm(all_stocks):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.info
                    data["Symbol"] = ticker
                    # .append to be depricated; but still going to use it
                    dataset = pd.concat([dataset, pd.DataFrame([data])])

                    # constantly writing as it is not exhaustive and in the cause of failure
                    dataset.to_csv("equities/all_stocks.csv")
                except:
                    pass


class EquityData(Equities):
    db_name = "database.db"

    def __init__(self):
        Equities.__init__(self)

    @staticmethod
    def batch(iterable, n=1):
        for ndx in range(0, len(iterable), n):
            yield iterable[ndx : min(ndx + n, len(iterable))]

    def download_ticker_data(
        self, stocks: list, interval: str = "1h", period: str = "5y"
    ):
        """
        yf.download(tickers = "SPY AAPL",  # list of tickers
            period = "1y",         # time period
            interval = "1d",       # trading interval
            prepost = False,       # download pre/post market hours data?
            repair = True)         # repair obvious price errors e.g. 100x?
        """

        # this sets up the longest potential query as intraday only goes back 2 years
        # may present an issue with number of samples
        stocks = list(stocks)
        if not any(x in interval.lower() for x in ["y", "m", "d"]):
            period = "2y"

        for batch_stocks in self.batch(stocks, 100):
            subset = dict()
            data = ydownload(
                batch_stocks, period=period, interval=interval, group_by="ticker"
            )
            for stock in batch_stocks:
                try:
                    subset[stock] = data[stock]
                except:
                    pass

            for key, df in subset.items():
                with sqlite3.connect(self.db_name) as conn:
                    table_name = f"{key}_{interval.lower()}"
                    df.to_sql(table_name, conn, if_exists="replace")

    def stock_data(self, interval: str = "1h") -> dict:
        """
        This will provide the actual data needed for a frequency
        :param interval:
        :return:
        """
        dataset = dict()
        for stock in self.consumer_stocks:
            dataset[stock] = self.get_data(stock, interval)
        return dataset

    def get_data(self, stock: str,
                 interval: str = '1h') -> pd.DataFrame:
        try:
            # this is a bit of a hard code; and the database should have been created
            # with appropriate index names, but this isn't a production case so who cares
            applied_index_name = 'index'
            if interval == '1d':
                applied_index_name = 'Date'

            table_name = f"{stock}_{interval.lower()}"
            query = f"SELECT * FROM [{table_name}]"
            with sqlite3.connect(self.db_name) as conn:
                df = pd.read_sql(query, con=conn, index_col=applied_index_name, parse_dates=True)
                df.index = [pd.Timestamp(x).replace(tzinfo=None) for x in df.index]
                return df
        except Exception as e:
            print(e)
            return pd.DataFrame()

    @property
    def consumer_stocks(self):
        # hardcoded
        if self._consumer_stocks is None:
            disc = pd.read_csv("equities/discretionary.csv")
            staples = pd.read_csv("equities/staples.csv")
            # again I will use .append to the end
            stocks = pd.concat([disc, staples])["Symbol"].to_numpy()
            self._consumer_stocks = stocks

        return self._consumer_stocks

    def update_data(self) -> pd.DataFrame:
        """
        Explicit for this use case
        :return: pd.DataFrame
        """

        stocks = self.consumer_stocks
        self.download_ticker_data(stocks, "1h")
        self.download_ticker_data(stocks, "1d")


# class SeekingAlpha(object):
#     # free tier API keys for now
#     _api_key = {
#         "X-RapidAPI-Key": "65d2e2853amsh4f2ae5552f99c88p13cfc5jsn3a29e2ed9956",
#         "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com",
#     }

#     def __init__(self):
#         pass
#         # self.analysis_titles()

#     def all_stocks(self, path: str = "equities/all_stocks.csv"):
#         """
#         Returns list of all stocks from specified CSV.
#         :param path:
#         :return self._all_stocks:
#         """
#         # Function called once so not yet assigned to class variable.
#         # limited to first 3 stocks while still in free tier
#         all_stocks = (
#             pd.read_csv(path, usecols=[0], engine="c").iloc[:, 0].to_list()[0:3]
#         )
#         return all_stocks

#     def get_analysis_meta(self, querystring: dict):
#         """
#         Yields paged JSON responses from /analysis/v2/list endpoint.
#         :param querystring:
#         :return get_analysis_meta generator:
#         """
#         url = "https://seeking-alpha.p.rapidapi.com/analysis/v2/list"
#         first_page = self._session.get(url, headers=self._api_key, params=querystring)
#         if first_page.status_code == 403:
#             print("API limit reached")
#             return
#         yield first_page.json()
#         num_pages = first_page.json()["meta"]["page"]["totalPages"]

#         for page in range(2, num_pages + 1):
#             querystring["number"] = str(page)
#             next_page = self._session.get(
#                 url, headers=self._api_key, params=querystring
#             ).json()
#             yield next_page

#     def process_meta_json(self, page: dict):
#         """
#         Transform JSON dicts from get_analysis_meta into DataFrame.
#         :param page:
#         :return dataset:
#         """
#         data = pd.DataFrame()
#         for article in page["data"]:
#             meta = {}
#             meta["article_id"] = [article.get("id")]
#             attributes = article.get("attributes", {})
#             relationships = article.get("relationships", {})
#             meta["title"] = [attributes.get("title")]
#             meta["timestamp"] = [attributes.get("publishOn")]
#             meta["link"] = [article.get("links", {}).get("self")]
#             meta["comments"] = [attributes.get("commentCount")]
#             meta["locked_pro"] = [attributes.get("isLockedPro")]
#             meta["paywalled"] = [attributes.get("isPaywalled")]
#             meta["author_id"] = [
#                 relationships.get("author", {}).get("data", {}).get("id")
#             ]
#             sentiments = relationships.get("sentiments", {}).get("data")
#             # The following attributes may not be needed in final version
#             meta["sentiment_ids"] = [[i["id"] for i in sentiments]] if not [] else [[]]
#             primary_tickers = relationships.get("primaryTickers", {}).get("data")
#             meta["primary_tickers"] = (
#                 [[i["id"] for i in primary_tickers]] if not [] else [[]]
#             )
#             secondary_tickers = relationships.get("secondaryTickers", {}).get("data")
#             meta["secondary_tickers"] = (
#                 [[i["id"] for i in secondary_tickers]] if not [] else [[]]
#             )
#             data = pd.concat([data, pd.DataFrame(meta)])
#         return data

#     def analysis_titles(self, path: str = "seeking_alpha_analysis.csv"):
#         """
#         Retrieves metadata DataFrame for all SA analyses of each ticker in self.all_stocks().
#         Writes DataFrame to CSV.
#         :param path:
#         :return dataset:
#         """
#         if not os.path.isfile(path):
#             self._session = requests.Session()
#             dataset = pd.DataFrame()
#             for ticker in tqdm(self.all_stocks()):
#                 try:
#                     querystring = {"id": ticker, "size": "40", "number": "1"}
#                     for page in self.get_analysis_meta(querystring):
#                         ticker_data = self.process_meta_json(page)
#                         ticker_data["ticker"] = ticker
#                         dataset = pd.concat([dataset, ticker_data])
#                         dataset.to_csv(path, index=False)
#                 except:
#                     pass
#         else:
#             dataset = pd.read_csv(path)
#         return dataset


class TwitterData(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_m_to_float(x):
        if type(x) == float or type(x) == int:
            return x
        return float(x.replace("M", "")) * 1000000

    def _sns_search(self, symbol: str, since: date, until: date):
        """
        Wrapper that retrieves top tweets (according to Twitter) for a given date range with
        the specified symbol's cashtag. The specific tweets retrieved can vary
        due to different date ranges.
        Uses the TwitterSearchScraper from snscrape to retrieve Tweets.
        https://github.com/JustAnotherArchivist/snscrape
        Returns tweets in a dataframe with two columns: ["date", "rawContent"]
        """
        print(
            f"Harvesting ${symbol}: {since.strftime('%Y-%m-%d')} to {until.strftime('%Y-%m-%d')}",
            end="\r",
        )
        # Assemble Twitter-friendly search string, with some constraints to reduce overall number of tweets
        search_str = f"${symbol} lang:en since:{since.strftime('%Y-%m-%d')} until:{until.strftime('%Y-%m-%d')} -filter:links -filter:replies"
        snstweet = TwitterSearchScraper(
            search_str,
            mode=TwitterSearchScraperMode.TOP,
        )
        # Tweet timestamps and contents are retrieved through a generator and stored in a DataFrame
        result = pd.DataFrame(
            [[tweet.date, tweet.rawContent] for tweet in snstweet.get_items()],
            columns=["date", "rawContent"],
        )
        return result

    def harvest_cashtag(
        self,
        symbol: str,
        start: date = date(2021, 6, 5),
        end: date = date.today(),
        overwrite: bool = True,
    ):
        """
        Harvests tweets for the pre-specified date range of a given symbol.
        """
        # Turn off logging to suppress snscrape output
        logging.disable(logging.CRITICAL)
        # Create empty .csv with headers for our symbol
        if overwrite:
            with open(f"tweets/{symbol}.csv", "w") as f:
                f.write('"timestamp","content","symbol"\n')
        since = start
        delta = timedelta(days=7)
        until = since + delta
        # Run weekly (Sat to Fri) search for top tweets in specified time range
        while since <= end:
            result = self._sns_search(symbol, since, until)
            current_min = pd.to_datetime(result["date"]).min()
            # Check if any tweets are missing by comparing earliest date in
            # `results` with our specified `since` date.
            # Twitter search returns tweets in descending order of timestamp,
            # so only the beginning of the date range needs to be checked.
            try:
                # While the earliest date does not match the specified `since` date,
                # run a search with a shortened time range (same `since`, earlier `until`)
                # to retrieve missing tweets (if there are any).
                while current_min.date() > since:
                    decreased_until = (current_min + timedelta(days=1)).date()
                    result = pd.concat(
                        [result, self._sns_search(symbol, since, decreased_until)]
                    )
                    # If reducing time range does not change results (which can be checked
                    # by comparing the new results' earliest timestamp to the previously-noted
                    # earliest timestamp), there may not be any additional tweets.
                    # If that is the case, the additional searches can cease.
                    if pd.to_datetime(result["date"]).min() == current_min:
                        break
                    # Minimum timestamp is updated after the comparison step of each iteration.
                    current_min = pd.to_datetime(result["date"]).min()
            except:
                pass
            # Multiple searches in same range may result in duplicates
            result = result[["date", "rawContent"]].drop_duplicates()
            # Convert timestamps to datetime.datetime
            result["date"] = pd.to_datetime(result["date"])
            # Remove excess line breaks and whitespace in tweet contents
            result["rawContent"] = result["rawContent"].str.replace(
                r"\r+|\n+|\t+", " ", regex=True
            )

            # Sort tweets by timestamp for this iteration's date range
            # Don't write header to file because the header is already written.
            # Final DataFrame (result) for each time period (iteration) is appended to the csv,
            # then `result` is overwritten to reduce memory usage and save progress.
            result["symbol"] = symbol
            result.sort_values(by="date").to_csv(
                f"tweets/{symbol}.csv",
                mode="a",
                index=False,
                header=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            # Increment twitter search date range
            since += delta
            until += delta
        # Turn logging back on
        logging.disable(logging.NOTSET)

    def update_cashtag(self, symbol: str):
        # If latest tweets in data are not current,
        # redo search for each missing week until present day and/or
        # redo search for current week beginning at the latest Saturday.
        # Method is rudimentary because it does not allow for an end date
        # that is not today nor does it update the data given an earlier start date.
        # Will add those options if necessary.
        tweets = pd.read_csv(f"tweets/{symbol}.csv")
        tweets["timestamp"] = pd.to_datetime(tweets["timestamp"])
        if tweets["timestamp"].max().date() >= date.today():
            return
        latest_saturday = (
            tweets[tweets["timestamp"].dt.dayofweek == 5]["timestamp"].max().date()
        )
        tweets = tweets[tweets["timestamp"].dt.date < latest_saturday]
        tweets.to_csv(f"tweets/{symbol}.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)
        self.harvest_cashtag(
            symbol=symbol,
            start=latest_saturday,
            overwrite=False,
        )

    def get_tweets(
        self, symbols: list = None, number: int = 100, refresh_cashtag=False
    ):
        # If no symbols are specified, function will retrieve tweets
        # for the top # of equities according to market cap (default = 100)
        if not symbols:
            # Retrieve consumer discretionaries and consumer staples from CSVs
            symbols = pd.concat(
                [
                    pd.read_csv("equities/discretionary.csv"),
                    pd.read_csv("equities/staples.csv"),
                ]
            )
            # Exclude equities with different classes or additional circumstances
            symbols = symbols[
                ~(
                    symbols["Symbol"].str.contains("/")
                    | (symbols["Symbol"].str.len() > 4)
                )
            ]

            # Transform `Market Cap` column into sortable floats and retrieve top # equities
            symbols["Market Cap"] = (
                symbols["Market Cap"]
                .str.replace(",", "")
                .apply(self.convert_m_to_float)
            )
            symbols = symbols.sort_values(by="Market Cap", ascending=False)[
                "Symbol"
            ].to_list()[:number]

        # Loop through `tweets` directory to check if CSV exists for each equity
        # Run self._harvest_cashtag (time-consuming) on missing equities
        if not os.path.exists("tweets"):
            os.makedirs("tweets")
        for symbol in symbols:
            if not os.path.isfile(f"tweets/{symbol}.csv"):
                self.harvest_cashtag(symbol=symbol)
                print("\n")
            elif refresh_cashtag:
                # Refresh any data that was not just retrieved.
                self.update_cashtag(symbol=symbol)
                print("\n")

        result = pd.DataFrame()
        # Retrieve data from CSVs
        for symbol in symbols:
            result = pd.concat([result, pd.read_csv(f"tweets/{symbol}.csv")])
        return result


if __name__ == "__main__":
    EquityData().update_data()
