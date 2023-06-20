# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:17:44 2023

@author: rkedia
"""

import praw
from dotenv import load_dotenv
import os
import re
from collections import defaultdict
from data import EquityData
from datetime import datetime, timedelta
import yaml
import pandas as pd


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords


# Load environment variables from .env file
load_dotenv()

# import pdb; pdb.set_trace()


class RedditAnalyzer:
    def __init__(self):
        # Initialize Reddit instance
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv("REDDIT_USER_AGENT")
        self.username = os.getenv("REDDIT_USERNAME")
        self.password = os.getenv("REDDIT_PASSWORD")

        # self.word_embeddings_model = gensim.models.Word2Vec.load('path_to_word_embeddings_model')
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.equity_data = EquityData()
        # self.stock_tickers = stocks
        self.stop_words = ["I", "AND", "THE"]

    @staticmethod
    def get_top_stock_symbols(stock_count=100):
        # universe of consumer discretionary and consumer staples stocks
        disc = pd.read_csv("equities/discretionary.csv")
        staples = pd.read_csv("equities/staples.csv")

        # concatenate into one dataframe
        stocks = pd.concat([disc, staples])[["Symbol", "Market Cap"]]

        # stocks only with 4 letters or less
        stocks = stocks[stocks["Symbol"].str.len() < 5]

        # replace 'M' with 6 zeros and cast string as float
        stocks["Market Cap"] = stocks["Market Cap"].apply(
            lambda x: float(x.replace(" M", "000000").replace(",", ""))
        )

        stocks = stocks.sort_values("Market Cap", ascending=False)[:stock_count]
        stocks = stocks["Symbol"].to_numpy()
        return stocks

    def tokenize(self, sentence):
        tokens = word_tokenize(sentence)
        return tokens

    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmas

    def remove_stopwords(self, tokens):
        stopwords_list = set(stopwords.words("english"))
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopwords_list
        ]
        return filtered_tokens

    def process_sentence(self, sentence):
        # tokenize
        tokens = self.tokenize(sentence)

        # lemmatize
        lemmas = self.lemmatize(tokens)

        # remove stopwords
        filtered_tokens = self.remove_stopwords(lemmas)

        # dictionary of sentiment score attributes for string of text
        sentiment_scores = self.vader_analyzer.polarity_scores(
            " ".join(filtered_tokens)
        )

        # retunrns just a single, compound sentiment score
        sentiment_score = sentiment_scores["compound"]

        filtered_tokens.append(sentiment_score)
        return filtered_tokens

    def generate_word_embeddings(self, sentence):
        # Generate word embeddings for a given sentence using the word embeddings model
        # You'll need to modify this based on the specific word embeddings library you're using
        word_embeddings = self.word_embeddings_model[sentence]
        return word_embeddings

    def retrieve_submission_text(self, subreddit_topic, timefilter="year"):
        # reddit instance
        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            username=self.username,
            password=self.password,
        )

        # start_time = datetime.utcnow() - timedelta(days=365 * 2)
        subreddit = reddit.subreddit(subreddit_topic)
        submissions = subreddit.hot(limit=10000)
        submission_data = []

        for submission in submissions:
            if "CLICK HERE FOR" not in submission.selftext:
                submission_data.append(
                    {
                        "date": datetime.fromtimestamp(submission.created_utc).strftime(
                            "%Y-%m-%d"
                        ),
                        "text": submission.selftext,
                    }
                )

        return submission_data
        # return [submission.selftext for submission in submissions if "CLICK HERE FOR" not in submission.selftext]

    def retrieve_tickers_sentences(self, stock_posts):
        # import pdb; pdb.set_trace()
        pattern = r"\b[A-Z]{1,5}\b"
        mydict = defaultdict(list)

        for post in stock_posts:
            sentences = re.split(r"(?<=[.?!])\s+", post["text"])
            # agg_posts = ' '.join([list(stock_post.values())[-1] for stock_post in stock_posts])
            # sentences = re.split(r'(?<=[.?!])\s+', agg_posts)
            for sentence in sentences:
                tickers = re.findall(pattern, sentence)

                sentence_tokens = self.process_sentence(sentence)
                sentiment_scores = sentence_tokens.pop()

                for ticker in tickers:
                    if ticker in RedditAnalyzer.get_top_stock_symbols():
                        mydict[ticker].append(
                            {"date": post["date"], "sentiment_score": sentiment_scores}
                        )
        return mydict


reddit_analyzer = RedditAnalyzer()
stock_posts = reddit_analyzer.retrieve_submission_text("stocks")
investing_posts = reddit_analyzer.retrieve_submission_text("investing")
wsb_posts = reddit_analyzer.retrieve_submission_text("wallstreetbets")

posts = stock_posts + investing_posts + wsb_posts

ticker_sentence_dict = reddit_analyzer.retrieve_tickers_sentences(posts)
# print(ticker_sentence_dict)


with open("reddit_scores.yaml", "w") as file:
    yaml.dump(ticker_sentence_dict, file)
