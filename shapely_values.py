import pandas as pd

from supervised_ensemble import Model

from helpers import merge_dataframes
from sklearn.ensemble import RandomForestRegressor


def create_datasets(m: Model):
    datasets = m.create_model_datasets()
    X, Y = m.create_X_Y(datasets)

    bollinger_X = m.create_objective(X.copy(), 'bollinger')
    sentiment_dataset = m.apply_sentiment_data(X.copy(), m.sentiment_dataset)
    sentiment_dataset.dropna(inplace=True)

    applied_dataset = m.apply_sentiment_data(bollinger_X.copy(), m.sentiment_dataset)
    applied_dataset.dropna(inplace=True)

    return {'basic_sentiment': sentiment_dataset,
            'combined_tehcnical': applied_dataset,
            'Y': Y}


def create_train_test(m: Model,
                      X: pd.DataFrame,
                      Y: pd.DataFrame):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    x_cols = X.columns
    y_cols = Y.columns

    dataset = merge_dataframes([X, Y]).dropna()
    X = pd.DataFrame(dataset[x_cols])
    Y = pd.DataFrame(dataset[y_cols])

    xtrain, xtest, ytrain, ytest = m.temporal_train_test_split(X, Y)
    xtrain = xtrain.loc[xtrain.index.isin(ytrain.index)]
    ytrain = ytrain.loc[ytrain.index.isin(xtrain.index)]
    xtest = xtest.loc[xtest.index.isin(ytest.index)]
    ytest = ytest.loc[ytest.index.isin(xtest.index)]

    return xtrain, xtest, ytrain, ytest


if __name__ == '__main__':
    m = Model()
    datasets = create_datasets(m)
    print(datasets)

    # CHOOSE whatever model; setting RF
    ensemble = RandomForestRegressor()

    xtrain, xtest, ytrain, ytest = create_train_test(m, datasets['basic_sentiment'], datasets['Y'])

    ensemble.fit(xtrain, ytrain)
    pred = pd.DataFrame(ensemble.predict(xtest), index=xtest.index)
