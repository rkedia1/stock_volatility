import pandas as pd

from ta.volatility import BollingerBands
from ta.trend import MACD


from data import EquityData, FinancialsData
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    from umap.umap_ import UMAP
except: pass
import matplotlib.pyplot as plt
import numpy as np


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


class ClusterFinancials(FinancialsData):
    cluster_features = [
        "DilutedEPS",
        "NormalizedEBITDA",
        "TotalRevenue",
        "MarketCap",
        "PriceYoY2021",
        "PriceYoY2022",
        "RevenueYoY2021",
        "RevenueYoY2022",
        "PeRatio",
    ]

    def __init__(self, sentiment: bool = False):
        FinancialsData.__init__(self)
        self.yf_data = self.get_yfinance_data()
        if sentiment:
            self.cluster_features.append('AvgSentiment2022')
        self.dimensions = StandardScaler().fit_transform(
            self.yf_data[self.cluster_features].values
        )
        self.labels = self.yf_data.reset_index().symbol.to_list()
        self.kmeans = self.KMeans(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
        self.agglomerativeclustering = self.AgglomerativeClustering(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
        self.dbscan = self.DBSCAN(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
        self.pca = self.PCA(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
        self.gmm = self.GMM(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
        self.umap = self.UMAP(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
        self.tsne = self.TSNE(
            yf_data=self.yf_data,
            cluster_features=self.cluster_features,
            dimensions=self.dimensions,
            labels=self.labels,
        )
    class KMeans:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels
            self.random_state = 42

        def inertia(
            self,
            max_clusters: int = 12,
            init: str = "k-means++",
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 0.0001,
            algorithm: str = "lloyd",
        ):
            within_group_sum_of_squares = []
            K = range(2, max_clusters + 1)
            for k in K:
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    algorithm=algorithm,
                    random_state=self.random_state,
                )
                km = km.fit(self.dimensions)
                within_group_sum_of_squares.append(km.inertia_)
            return K, within_group_sum_of_squares

        def calinski_harabasz(
            self,
            max_clusters: int = 12,
            init: str = "k-means++",
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 0.0001,
            algorithm: str = "lloyd",
        ):
            calinski_harabasz_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    algorithm=algorithm,
                    random_state=self.random_state,
                )
                km = km.fit(self.dimensions)
                calinski_harabasz_scores.append(
                    calinski_harabasz_score(self.dimensions, km.labels_)
                )
            return K, calinski_harabasz_scores

        def silhouette(
            self,
            max_clusters: int = 12,
            init: str = "k-means++",
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 0.0001,
            algorithm: str = "lloyd",
            scoring_metric: str = "euclidean",
        ):
            silhouette_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    algorithm=algorithm,
                    random_state=self.random_state,
                )
                km = km.fit(self.dimensions)
                silhouette_scores.append(
                    silhouette_score(self.dimensions, km.labels_, metric=scoring_metric)
                )
            return K, silhouette_scores

        def davis_bouldin(
            self,
            max_clusters: int = 12,
            init: str = "k-means++",
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 0.0001,
            algorithm: str = "lloyd",
        ):
            davis_bouldin_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    algorithm=algorithm,
                    random_state=self.random_state,
                )
                km = km.fit(self.dimensions)
                davis_bouldin_scores.append(
                    davies_bouldin_score(self.dimensions, km.labels_)
                )
            return K, davis_bouldin_scores

        def cluster(
            self,
            n_clusters: int = 3,
            init: str = "k-means++",
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 0.0001,
            algorithm: str = "lloyd",
        ):
            km = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                algorithm=algorithm,
                random_state=self.random_state,
            )
            km = km.fit(self.dimensions)
            return self.labels, km

    class AgglomerativeClustering:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels

        def calinski_harabasz(
            self,
            max_clusters: int = 12,
            metric: str = "euclidean",
            connectivity: str = None,
            linkage: str = "ward",
            compute_distances: bool = False,
        ):
            calinski_harabasz_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                ac = AgglomerativeClustering(
                    n_clusters=k,
                    metric=metric,
                    connectivity=connectivity,
                    linkage=linkage,
                    compute_distances=compute_distances,
                )
                ac = ac.fit(self.dimensions)
                calinski_harabasz_scores.append(
                    calinski_harabasz_score(self.dimensions, ac.labels_)
                )
            return K, calinski_harabasz_scores

        def silhouette(
            self,
            max_clusters: int = 12,
            metric: str = "euclidean",
            connectivity: str = None,
            linkage: str = "ward",
            compute_distances: bool = False,
            scoring_metric: str = "euclidean",
        ):
            silhouette_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                ac = AgglomerativeClustering(
                    n_clusters=k,
                    metric=metric,
                    connectivity=connectivity,
                    linkage=linkage,
                    compute_distances=compute_distances,
                )
                ac = ac.fit(self.dimensions)
                silhouette_scores.append(
                    silhouette_score(self.dimensions, ac.labels_, metric=scoring_metric)
                )
            return K, silhouette_scores

        def davis_bouldin(
            self,
            max_clusters: int = 12,
            metric: str = "euclidean",
            connectivity: str = None,
            linkage: str = "ward",
            compute_distances: bool = False,
        ):
            davis_bouldin_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                ac = AgglomerativeClustering(
                    n_clusters=k,
                    metric=metric,
                    connectivity=connectivity,
                    linkage=linkage,
                    compute_distances=compute_distances,
                )
                ac = ac.fit(self.dimensions)
                davis_bouldin_scores.append(
                    davies_bouldin_score(self.dimensions, ac.labels_)
                )
            return K, davis_bouldin_scores

        def cluster(
            self,
            n_clusters: int = 3,
            metric: str = "euclidean",
            connectivity: str = None,
            linkage: str = "ward",
            compute_distances: bool = False,
        ):
            ac = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                connectivity=connectivity,
                linkage=linkage,
                compute_distances=compute_distances,
            )
            ac = ac.fit(self.dimensions)
            return self.labels, ac

    class DBSCAN:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels

        def calinski_harabasz(
            self,
            eps: float = 0.5,
            min_samples: int = 5,
            metric: str = "euclidean",
            algorithm: str = "auto",
            leaf_size: int = 30,
            p: float = None,
            n_jobs: int = None,
        ):
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=n_jobs,
            )
            db = db.fit(self.dimensions)
            return calinski_harabasz_score(self.dimensions, db.labels_)

        def silhouette(
            self,
            eps: float = 0.5,
            min_samples: int = 5,
            metric: str = "euclidean",
            algorithm: str = "auto",
            leaf_size: int = 30,
            p: float = None,
            n_jobs: int = None,
            scoring_metric: str = "euclidean",
        ):
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=n_jobs,
            )
            db = db.fit(self.dimensions)
            return silhouette_score(self.dimensions, db.labels_, metric=scoring_metric)

        def davis_bouldin(
            self,
            eps: float = 0.5,
            min_samples: int = 5,
            metric: str = "euclidean",
            algorithm: str = "auto",
            leaf_size: int = 30,
            p: float = None,
            n_jobs: int = None,
        ):
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=n_jobs,
            )
            db = db.fit(self.dimensions)
            return davies_bouldin_score(self.dimensions, db.labels_)

        def cluster(
            self,
            eps: float = 0.5,
            min_samples: int = 5,
            metric: str = "euclidean",
            algorithm: str = "auto",
            leaf_size: int = 30,
            p: float = None,
            n_jobs: int = None,
        ):
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=n_jobs,
            )
            db = db.fit(self.dimensions)
            return self.labels, db

    class GMM:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels
            self.random_state = 42

        def cluster(
            self,
            max_iter: int = 100,
            n_components: int = 4,
        ):
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=self.random_state,
                max_iter=max_iter,
            ).fit(self.dimensions)
            predicted_labels = gmm.predict(self.dimensions)
            groups = zip(self.labels, predicted_labels)
            return groups

        def cluster_probabilities(self, max_iter: int = 100):
            gmm = GaussianMixture(
                n_components=len(self.cluster_features),
                random_state=self.random_state,
                max_iter=max_iter,
            ).fit(self.dimensions)
            probabilities = gmm.predict_proba(self.dimensions)
            return probabilities

        def ideal_cluster_num(self, max_clusters: int = 10):
            bics = []
            for i in range(1, max_clusters + 1):
                gmm = GaussianMixture(
                    n_components=i,
                    random_state=self.random_state,
                    max_iter=100,
                ).fit(self.dimensions)
                bics.append(gmm.bic(self.dimensions))
            optimal_clust = np.argmin(bics) + 1
            return optimal_clust

    class PCA:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels
            self.random_state = 42

        def reduce(
            self,
            n_components: int = 2,
        ):
            pca_nd = PCA(n_components=n_components, random_state=self.random_state)
            results = pca_nd.fit_transform(self.dimensions)
            return results

    class UMAP:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels

        def reduce(
            self,
            n_neighbors: int = 15,
            min_dist: int = 0.1,
            n_components: int = 2,
            metric: str = "euclidean",
        ):
            fit = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
            )
            embedding = fit.fit_transform(self.dimensions)
            return embedding

    class TSNE:
        def __init__(self, yf_data, cluster_features, dimensions, labels):
            self.yf_data = yf_data
            self.cluster_features = cluster_features
            self.dimensions = dimensions
            self.labels = labels
            self.random_state = 42

        def reduce(
            self,
            n_components: int = 2,
            verbose: int = 0,
            perplexity: int = 40,
            n_iter: int = 300,
        ):
            tsne = TSNE(
                n_components=n_components,
                verbose=verbose,
                perplexity=perplexity,
                n_iter=n_iter,
            )
            results = tsne.fit_transform(self.dimensions)
            return results

if __name__ == "__main__":
    # TODO sys.argv
    dataset = AnalysisTargets().create_target_dataset("PG")
