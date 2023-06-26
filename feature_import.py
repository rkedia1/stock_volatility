
import pandas as pd

from supervised_ensemble import Model

from helpers import merge_dataframes
from sklearn.ensemble import RandomForestRegressor

import shap
import numpy as np


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

    ensemble = RandomForestRegressor()

    X = datasets['basic_sentiment'].copy()
    X = X[['Close', 'VADER', 'Absolute Score']]

    xtrain, xtest, ytrain, ytest = create_train_test(m, X, datasets['Y'])

    ensemble.fit(xtrain, ytrain)

    imp = {key: val for key, val in zip(xtest.columns, ensemble.feature_importances_)}



    pred = pd.DataFrame(ensemble.predict(xtest), index=xtest.index)

    import altair as alt
    import pandas as pd

    # Your data as a dictionary
    data_dict = {
        'Metric': ['Close', 'VADER', 'finBERT'],
        'Value': [0.4291923092612075, 0.25396294727191576, 0.31684474346687674]
    }

    # Define your color palette, you can use any valid matplotlib colors
    colors = ['blue', 'green', 'yellow']


    # Convert the dictionary to a pandas DataFrame
    data_df = pd.DataFrame(data_dict)

    # Create a bar chart in Altair
    import matplotlib.pyplot as plt

    # Define your color palette, you can use any valid matplotlib colors
    colors = ['blue', 'orange', 'green']

    # Create a new figure and set its size
    plt.figure(figsize=(10, 6))

    # Plot the data as a bar chart, with each bar a different color
    plt.bar(data_dict['Metric'], data_dict['Value'], color=colors)

    # Remove the x label by setting it to an empty string
    plt.xlabel('')

    # Provide a label for the y-axis, and a title for the chart
    plt.ylabel('Value')
    plt.title('Feature Importance')

    # Save the figure as a PNG file
    plt.savefig('test.png')

    # Display the plot
    plt.show()
    chart = alt.Chart(data_df).mark_bar().encode(
        x='Metric',
        y='Value'
    )
