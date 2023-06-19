import pandas as pd

from ta.volatility import BollingerBands
from ta.trend import MACD


from data import EquityData, TechnicalClusteringData
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


class AnalysisTargets(EquityData):
    base_datasets = dict()

    def __init__(self):
        EquityData.__init__(self)

    def create_target_dataset(
        self, equity: str, deviation_window: int = 2, rolling_change_window: int = 2
    ):
        """

        :param equity: the stock in question
        :param deviation_window: interval in days to observe the standard deviation of the intraday percentage change
                                 should be an indicator of the intraday volatility observed in trading patterns
        :param rolling_change_window: a string representing the period of time to look at price change;
                                      will default to 2 days
        :return:
        """
        # creates a name to avoid having to pull the same data multiple times
        try:
            datasets = dict()
            intervals = ["1h", "1d"]
            for interval in intervals:
                name = f"{equity}_{interval}".lower()
                if name not in self.base_datasets:
                    self.base_datasets[name] = self.get_data(equity, interval)

                try:
                    df = self.base_datasets[name].copy()
                    if interval == "1h":
                        datasets["Deviation"] = self.create_intraday_dataset(
                            df, deviation_window
                        )
                    elif interval == "1d":
                        datasets["Base"] = self.base_datasets[name].copy()
                except Exception as e:
                    print(e)

            if all(x in datasets for x in ["Deviation", "Base"]):
                applied_dataset = datasets["Base"].copy()
                applied_dataset["Rolling Change"] = applied_dataset["Close"].pct_change(
                    rolling_change_window
                )
                applied_dataset = pd.merge(
                    applied_dataset,
                    datasets["Deviation"],
                    left_index=True,
                    right_index=True,
                )
                return applied_dataset
        except Exception as e:
            print(e)

    def create_intraday_dataset(self, df: pd.DataFrame, deviation_window: int):
        try:
            df["pct_change"] = df["Close"].pct_change() * 100

            df.dropna(inplace=True)
            df.index = pd.DatetimeIndex(df.index)

            # in the case the below doesn't make sense
            # https://stackoverflow.com/questions/24875671/resample-in-a-rolling-window-using-pandas
            rolling_std = pd.DataFrame(
                df[["pct_change"]]
                .rolling(f"{deviation_window}d")
                .std()
                .resample("1d")
                .first()
            )
            rolling_std.columns = ["Deviation"]
            return rolling_std
        except Exception as e:
            print(e)


class KmeansTechnicals(TechnicalClusteringData):
    random_state = 42

    def __init__(self):
        TechnicalClusteringData.__init__(self)
        self.yf_data = self.get_yfinance_data()

    def inertia(self, max_clusters: int = 21):
        labels = self.yf_data.index.to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        within_group_sum_of_squares = []
        K = range(2, max_clusters + 1)
        for k in K:
            km = KMeans(n_clusters=k, random_state=self.random_state, init="k-means++")
            km = km.fit(values)
            within_group_sum_of_squares.append(km.inertia_)
        return K, within_group_sum_of_squares

    def calinski_harabasz(self, max_clusters: int = 21):
        labels = self.yf_data.index.to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        calinski_harabasz_scores = []
        K = range(2, max_clusters + 1)
        for k in K:
            km = KMeans(n_clusters=k, random_state=self.random_state, init="k-means++")
            km = km.fit(values)
            calinski_harabasz_scores.append(calinski_harabasz_score(values, km.labels_))
        return K, calinski_harabasz_scores

    def silhouette(self, max_clusters: int = 21):
        labels = self.yf_data.index.to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        for k in K:
            km = KMeans(n_clusters=k, random_state=self.random_state, init="k-means++")
            km = km.fit(values)
            silhouette_scores.append(
                silhouette_score(values, km.labels_, metric="euclidean")
            )
        return K, silhouette_scores

    def davis_bouldin(self, max_clusters: int = 21):
        labels = self.yf_data.symbol.to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        davis_bouldin_scores = []
        K = range(2, max_clusters + 1)
        for k in K:
            km = KMeans(n_clusters=k, random_state=self.random_state, init="k-means++")
            km = km.fit(values)
            davis_bouldin_scores.append(davies_bouldin_score(values, km.labels_))
        return K, davis_bouldin_scores

    def cluster(self, num_clusters=3):
        labels = self.yf_data["symbol"].to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        km = KMeans(
            n_clusters=num_clusters, random_state=self.random_state, init="k-means++"
        )
        km = km.fit(values)
        groups = zip(labels, km.labels_)
        return groups


class GMMTechnicals(TechnicalClusteringData):
    random_state = 42

    def __init__(self):
        TechnicalClusteringData.__init__(self)
        self.yf_data = self.get_yfinance_data()

    def cluster(
        self,
        max_iter: int = 100,
    ):
        labels = self.yf_data["symbol"].to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        gmm = GaussianMixture(
            n_components=4, random_state=self.random_state, max_iter=max_iter
        ).fit(values)
        predicted_labels = gmm.predict(values)
        groups = zip(labels, predicted_labels)
        return groups


class TSNETechnicals(TechnicalClusteringData):
    random_state = 42

    def __init__(self):
        TechnicalClusteringData.__init__(self)
        self.yf_data = self.get_yfinance_data()

    def cluster(self):
        labels = self.yf_data["symbol"].to_list()
        values = self.yf_data[
            ["DilutedEPS", "NormalizedEBITDA", "TotalRevenue", "MarketCap"]
        ].values
        values = StandardScaler().fit_transform(values)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        results = tsne.fit_transform(values)
        return results


if __name__ == "__main__":
    # TODO sys.argv
    dataset = AnalysisTargets().create_target_dataset("PG")
