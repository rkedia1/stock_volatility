"""
@author Evan McFall
"""

import csv
import logging
import os
import sqlite3
import warnings
from datetime import date, timedelta

import nltk
import numpy as np
import pandas as pd
import requests
import torch
import yahooquery as yq
import yfinance as yf
from forex_python.converter import CurrencyRates
from nltk.sentiment import SentimentIntensityAnalyzer
from snscrape.modules.twitter import TwitterSearchScraper, TwitterSearchScraperMode
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from yfinance import download as ydownload

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon")


class Equities(object):
    _equities = None

    stock_filename = "equities/all_stocks.csv"
    sector_filename = "equities/sector.csv"

    # TODO secrets if for use in future
    finnhub_api_key = "########################"

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

    def get_data(self, stock: str, interval: str = "1h") -> pd.DataFrame:
        try:
            # # this is a bit of a hard code; and the database should have been created
            # # with appropriate index names, but this isn't a production case so who cares
            # applied_index_name = 'Datetime'
            # if interval == '1d':
            #     applied_index_name = 'Date'

            table_name = f"{stock}_{interval.lower()}"
            query = f"SELECT * FROM [{table_name}]"
            with sqlite3.connect(self.db_name) as conn:
                df = pd.read_sql(query, con=conn)
                # this should fix any version issues with sqlite; hotfix
                for col in ["index", "Datetime", "Date"]:
                    try:
                        df.set_index(col, inplace=True)
                    except:
                        pass
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

    def get_symbol_list(self, number: int = 99):
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
                | (symbols["Description"].str.contains("ETF"))
            )
        ]

        # Transform `Market Cap` column into sortable floats and retrieve top # equities
        symbols["Market Cap"] = (
            symbols["Market Cap"].str.replace(",", "").apply(self.convert_m_to_float)
        )
        symbols = symbols.sort_values(by="Market Cap", ascending=False)[
            "Symbol"
        ].to_list()[:number]
        return symbols

    def get_tweets(self, symbols: list = None, number: int = 99, refresh_cashtag=False):
        # If no symbols are specified, function will retrieve tweets
        # for the top # of equities according to market cap (default = 99)
        if not symbols:
            # Retrieve consumer discretionaries and consumer staples from CSVs
            symbols = self.get_symbol_list()
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

    def sentiment_vader(self, folder: str = "tweets-with-sentiment-VADER"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        tweets = self.get_tweets()
        for symbol in tweets["symbol"].unique():
            # import pdb; pdb.set_trace()
            if not os.path.exists(f"{folder}/{symbol}.csv"):
                ticker = tweets[tweets["symbol"] == symbol]
                analyzer = SentimentIntensityAnalyzer()
                ticker["sentiment_score"] = ticker["content"].apply(
                    lambda x: analyzer.polarity_scores(x)["compound"]
                )
                ticker.to_csv(
                    f"{folder}/{symbol}.csv", quoting=csv.QUOTE_NONNUMERIC, index=False
                )

    def sentiment_vader_substitution(
        self, folder: str = "tweets-with-sentiment-VADER-substituted"
    ):
        if not os.path.exists(folder):
            os.makedirs(folder)
        tweets = self.get_tweets()
        for symbol in tweets["symbol"].unique():
            # import pdb; pdb.set_trace()
            if not os.path.exists(f"{folder}/{symbol}.csv"):
                ticker = tweets[tweets["symbol"] == symbol]
                ticker["content"] = ticker["content"].str.replace(
                    "bought puts", "doubt", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "buying puts", "doubting", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "bought calls", "trust", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "buying calls", "trusting", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "call", "positive", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "put", "negative", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "long", "positive", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "shorting", "negative", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "short", "negative", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "bullish", "optimistic", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "bull", "optimist", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "bearish", "pessimistic", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "bear", "pessimist", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "buying", "positive", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "buy", "positive", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "bought", "positive", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "up", "positive", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "down", "negative", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "-", "negative ", case=False
                )
                ticker["content"] = ticker["content"].str.replace(
                    "+", "positive ", case=False
                )
                analyzer = SentimentIntensityAnalyzer()
                ticker["sentiment_score"] = ticker["content"].apply(
                    lambda x: analyzer.polarity_scores(x)["compound"]
                )
                ticker.to_csv(
                    f"{folder}/{symbol}.csv", quoting=csv.QUOTE_NONNUMERIC, index=False
                )

    def average_sentiment(
        self, symbol: str, start=date(2022, 1, 1), end=date(2022, 12, 31)
    ):
        if not os.path.exists("tweets-with-sentiment-VADER"):
            self.sentiment_vader()
        tweets_with_sentiment = pd.read_csv(f"tweets-with-sentiment-VADER/{symbol}.csv")
        tweets_with_sentiment["timestamp"] = pd.to_datetime(
            tweets_with_sentiment["timestamp"]
        )
        tweets_with_sentiment = tweets_with_sentiment[
            (tweets_with_sentiment["timestamp"].dt.date >= start)
            & (tweets_with_sentiment["timestamp"].dt.date <= end)
        ]
        average_sentiment = tweets_with_sentiment["sentiment_score"].mean()
        return average_sentiment

    def sentiment_finbert(self, folder: str = "tweets-with-sentiment-finBERT"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        symbols = self.get_symbol_list()
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        curr_step = 0
        for symbol in symbols:
            curr_step += 1
            if not os.path.exists(f"{folder}/{symbol}.csv"):
                print(f"[{curr_step}/{len(symbols)}] {symbol}:")
                tweets = pd.read_csv(f"tweets/{symbol}.csv")
                contents = tweets.values
                df = pd.DataFrame()
                for i in tqdm(contents):
                    inputs = tokenizer(
                        i[1], padding=True, truncation=True, return_tensors="pt"
                    )
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    positive = predictions[:, 0].tolist()
                    negative = predictions[:, 1].tolist()
                    neutral = predictions[:, 2].tolist()

                    table = {
                        "symbol": i[2],
                        "timestamp": i[0],
                        "content": i[1],
                        "Positive": positive,
                        "Negative": negative,
                        "Neutral": neutral,
                    }

                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                table,
                                columns=[
                                    "symbol",
                                    "timestamp",
                                    "content",
                                    "Positive",
                                    "Negative",
                                    "Neutral",
                                ],
                            ),
                        ]
                    )

                df.to_csv(
                    f"{folder}/{symbol}.csv", quoting=csv.QUOTE_NONNUMERIC, index=False
                )


class FinancialsData(object):
    def __init__(self, symbols=None):
        self.symbols = self.get_symbols_mktcaps()

    def get_symbols_mktcaps(self):
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
                | (symbols["Description"].str.contains("ETF"))
            )
        ]

        # Transform `Market Cap` column into sortable floats and retrieve top # equities
        symbols["Market Cap"] = (
            symbols["Market Cap"]
            .str.replace(",", "")
            .apply(TwitterData().convert_m_to_float)
        )
        symbols = (
            symbols.sort_values(by="Market Cap", ascending=False)[
                ["Symbol", "Market Cap", "Description"]
            ]
            .head(99)
            .values
        )
        return symbols

    def get_yfinance_data(self, refresh: bool = False):
        if not os.path.exists("symbol_fundamentals.csv") or refresh:
            cr = CurrencyRates()
            result = pd.DataFrame()
            for symbol, mktcap, description in self.symbols:
                if symbol != "ONON" and symbol != "RIVN" and symbol != "PSNY":
                    ticker = yq.Ticker(symbol)
                    info = ticker.get_financial_data(
                        types=["DilutedEPS", "NormalizedEBITDA", "TotalRevenue"],
                        trailing=False,
                    )
                    info_temp = info[info["asOfDate"].dt.year == 2021].sort_values(
                        by="asOfDate", ascending=False
                    )
                    if info_temp["asOfDate"].iloc[0].month < 3:
                        info_temp = info[info["asOfDate"].dt.year == 2022].sort_values(
                            by="asOfDate", ascending=True
                        )
                    info = info_temp
                    info["MarketCap"] = mktcap
                    if info["currencyCode"].iloc[0] != "USD":
                        info["DilutedEPS"].iloc[0] = round(
                            cr.convert(
                                info["currencyCode"].iloc[0],
                                "USD",
                                info["DilutedEPS"].iloc[0],
                                info["asOfDate"].iloc[0],
                            ),
                            2,
                        )
                        info["NormalizedEBITDA"].iloc[0] = round(
                            cr.convert(
                                info["currencyCode"].iloc[0],
                                "USD",
                                info["NormalizedEBITDA"].iloc[0],
                                info["asOfDate"].iloc[0],
                            ),
                            2,
                        )
                        info["TotalRevenue"].iloc[0] = round(
                            cr.convert(
                                info["currencyCode"].iloc[0],
                                "USD",
                                info["TotalRevenue"].iloc[0],
                                info["asOfDate"].iloc[0],
                            ),
                            2,
                        )
                        info["currencyCode"].iloc[0] = "USD"
                    price_change = ticker.history(
                        period="2y", interval="3mo", start="01-01-2021", adj_ohlc=True
                    )
                    info["PriceYoY2021"] = (
                        (
                            price_change.loc[(symbol, date(2022, 1, 1)), "open"]
                            - price_change.loc[(symbol, date(2021, 1, 1)), "open"]
                        )
                        / (price_change.loc[(symbol, date(2021, 1, 1)), "open"])
                    ) * 100
                    info["PriceYoY2022"] = (
                        (
                            price_change.loc[(symbol, date(2023, 1, 1)), "open"]
                            - price_change.loc[(symbol, date(2022, 1, 1)), "open"]
                        )
                        / (price_change.loc[(symbol, date(2022, 1, 1)), "open"])
                    ) * 100

                    revenue_change = ticker.income_statement(trailing=False)
                    info["RevenueYoY2021"] = (
                        (
                            revenue_change.iloc[-2].loc["TotalRevenue"]
                            - revenue_change.iloc[-3].loc["TotalRevenue"]
                        )
                        / (revenue_change.iloc[-3].loc["TotalRevenue"])
                    ) * 100
                    info["RevenueYoY2022"] = (
                        (
                            revenue_change.iloc[-1].loc["TotalRevenue"]
                            - revenue_change.iloc[-2].loc["TotalRevenue"]
                        )
                        / (revenue_change.iloc[-2].loc["TotalRevenue"])
                    ) * 100
                    price = ticker.history(start=info.iloc[0]["asOfDate"])
                    price = price.iloc[0]["close"]
                    info["PeRatio"] = price / info["DilutedEPS"]
                    prof = ticker.asset_profile
                    info["Description"] = description
                    info["Sector"] = prof[symbol]["sector"]
                    info["Industry"] = prof[symbol]["industry"]
                    info["AvgSentiment2022"] = TwitterData().average_sentiment(
                        symbol=symbol
                    )
                    result = pd.concat([result, info])
            result = result[
                [
                    "Description",
                    "Sector",
                    "Industry",
                    "asOfDate",
                    "DilutedEPS",
                    "NormalizedEBITDA",
                    "TotalRevenue",
                    "MarketCap",
                    "PriceYoY2021",
                    "PriceYoY2022",
                    "RevenueYoY2021",
                    "RevenueYoY2022",
                    "PeRatio",
                    "AvgSentiment2022",
                ]
            ]
            result.to_csv("symbol_fundamentals.csv")
        else:
            result = pd.read_csv("symbol_fundamentals.csv")
        return result


if __name__ == "__main__":
    EquityData().update_data()
    TwitterData().get_tweets()
    TwitterData().sentiment_finbert()
    TwitterData().sentiment_vader()
    TwitterData().sentiment_vader_substitution()
    FinancialsData().get_yfinance_data()
