# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:17:44 2023

@author: kedia
"""

import praw
from dotenv import load_dotenv
import os
import re
from collections import defaultdict
from data import EquityData

# Load environment variables from .env file
load_dotenv()


class RedditAnalyzer:
    def __init__(self):
        # Initialize Reddit instance
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv("REDDIT_USER_AGENT")
        self.username = os.getenv("REDDIT_USERNAME")
        self.password = os.getenv("REDDIT_PASSWORD")
        
        self.equity_data = EquityData()
        self.stock_tickers = self.equity_data.consumer_stocks
        self.stop_words = ['I', 'AND', 'THE']

    def retrieve_submission_text(self, subreddit_topic, count=500):
        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            username=self.username,
            password=self.password
        )
        subreddit = reddit.subreddit(subreddit_topic)
        submissions = subreddit.hot(limit=count)
        return [submission.selftext for submission in submissions if "CLICK HERE FOR" not in submission.selftext]

    def retrieve_tickers_sentences(self, stock_posts):
        mydict = defaultdict(list)
        pattern = r"\b[A-Z]{1,5}\b"
        stock_posts = ' '.join(stock_posts)
        sentences = re.split(r'(?<=[.?!])\s+', stock_posts)
        for sentence in sentences:
            tickers = re.findall(pattern, sentence)
            for ticker in tickers:
                if ticker in self.stock_tickers:
                    mydict[ticker].append(sentence)
        return mydict


reddit_analyzer = RedditAnalyzer()
stock_posts = reddit_analyzer.retrieve_submission_text("stocks")
investing_posts = reddit_analyzer.retrieve_submission_text("investing")
wsb_posts = reddit_analyzer.retrieve_submission_text("wallstreetbets")

posts = stock_posts + investing_posts + wsb_posts

ticker_sentence_dict = reddit_analyzer.retrieve_tickers_sentences(posts)
print(ticker_sentence_dict)
