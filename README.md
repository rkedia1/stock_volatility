# Predicting Equity Volatilty using Sentiment and Technical Analysis

## To set up a basic environment it should be pretty simple
## Run '__main__.py', which calls EquityData from data.py to check if the database exists and the needed files are there.
##### Made some of the instances relative imports to reduce project dependencies


### You can access the data via the CSVs (at the moment) and the main class
###### eg. from data import EquityData
### EquityData().stock_data(self, interval: str = '1h') -> pd.DataFrame:

### Must install latest developmental version of snscrape for Twitter scraper to work properly
###### eg. pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git
