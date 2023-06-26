import pickle

import pandas as pd

params = pd.read_csv('hyper.csv')
params.sort_values(by='score', inplace=True)

groups = pd.read_csv('stock_clusters.csv')
group_dict = groups.groupby('cluster')['symbol'].apply(list).to_dict()

from lightgbm import LGBMRegressor

ensemble = LGBMRegressor()

from supervised_ensemble import Model

m = Model()


def create_datasets(m: Model, equities: list or None = None):
    ensemble = LGBMRegressor()
    datasets = m.create_model_datasets(equities=equities)
    X, Y = m.create_X_Y(datasets)

    baseline_results = m.apply_model(ensemble, 'baseline', X.copy(), Y.copy())

    bollinger_X = m.create_objective(X.copy(), 'bollinger')
    bollinger_results = m.apply_model(ensemble, 'technical', bollinger_X, Y.copy())

    sentiment_dataset = m.apply_sentiment_data(X.copy(), m.sentiment_dataset)
    sentiment_dataset.dropna(inplace=True)
    sentiment_results = m.apply_model(ensemble, 'sentiment', sentiment_dataset, Y)

    applied_dataset = m.apply_sentiment_data(bollinger_X.copy(), m.sentiment_dataset)
    applied_dataset.dropna(inplace=True)
    combined_results = m.apply_model(ensemble, 'combined', applied_dataset, Y)

    dataset = {'baseline': baseline_results,
               'technical': bollinger_results,
               'sentiment': sentiment_results,
               'combined': combined_results}
    return dataset

baseline = create_datasets(m)

results = dict()
for i, group in group_dict.items():
    results[i] = create_datasets(m, group)

results['base'] = baseline

import pickle
with open('clustered_results.pkl', 'wb') as handle:
    pickle.dump(results, handle)

dataset = pd.DataFrame()
for key, val in results.items():
    df = pd.DataFrame(pd.DataFrame(val).iloc[1]).T
    df.index = [key]
    dataset = dataset.append(df)

