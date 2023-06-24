# @author Evan McFall


import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import TimeSeriesSplit

def err_handle(e: Exception,
               file: str):
    '''
    Creates a error log traceroute for the line number and file the error occured in
    :param e: Exception raised by sourcecode
    :param file: __file__ passback of the code
    :return: string of the error traceroute
    '''
    import sys

    try:
        base_tuple = ('Error on line {} in %s'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        file = file.split('/')[-1]
        return str(base_tuple) % file
    except Exception as ErrorHandle:
        print('err handle could not determine error')
        print(e)
        print(str(ErrorHandle) + str(__file__))


def merge_dataframes(list_of_dfs: list,
                     drop_duplicates: bool = False,
                     force_index: bool = False) -> pd.DataFrame:
    '''
    An easy merge of a list of different dataframes
    '''
    try:
        results = pd.DataFrame()  # what we will return
        if force_index:
            index_lengths = [len(df.index) for df in list_of_dfs]
            primary_index = list_of_dfs[index_lengths.index(min(index_lengths))].index
            min_index, max_index = min(primary_index), max(primary_index)

        for df in list_of_dfs:
            # we cannot pd.concat on this list; we will check if result exists
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if force_index:
                df = df.loc[(df.index >= min_index) & (df.index <= max_index)]

            if not results.empty:
                if drop_duplicates:
                    df = df[[x for x in df.columns if x not in results.columns]]
                results = pd.merge(results, df,
                                   how='outer',
                                   left_index=True,
                                   right_index=True)
            else:  # otherwise we set the dataframe as the result
                results = df
        return results
    except Exception as e:
        print(err_handle(e, __file__))


def remove_duplicate_index(df: pd.DataFrame,
                           priority: str = 'first') -> pd.DataFrame:
    if priority not in ['first', 'last']:
        raise AttributeError(priority, 'Not in first or last')
    return df.loc[~df.index.duplicated(keep=priority)]


def time_series_split(X: pd.DataFrame, Y: pd.DataFrame, n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits)
    X = X.reset_index()
    Y = Y.reset_index()

    X[["year", "month", "day", "ticker"]] = X['index'].apply(lambda x: pd.Series(str(x).split("_")))
    Y[["year", "month", "day", "ticker"]] = Y['index'].apply(lambda x: pd.Series(str(x).split("_")))
    X['date'] = pd.to_datetime(X[['year', 'month', 'day']])
    Y['date'] = pd.to_datetime(Y[['year', 'month', 'day']])

    X = X.set_index("index")
    Y = Y.set_index("index")
    X = X.sort_values(by=["date", "ticker"])
    Y = Y.sort_values(by=["date", "ticker"])
    dates = X['date'].unique()
    for train_index, test_index in tscv.split(dates):
        # MODIFICATION
        # train_date_start = dates[train_index[0]].date()
        train_date_start = pd.Timestamp(dates[train_index[0]])
        # train_date_end = dates[train_index[-1]].date()
        train_date_end = pd.Timestamp(dates[train_index[-1]])
        # test_date_start = dates[test_index[0]].date()
        test_date_start = pd.Timestamp(dates[test_index[0]])
        # test_date_end = dates[test_index[-1]].date()
        test_date_end = pd.Timestamp(dates[test_index[-1]])

        X_train = X[(X['date'].dt.date >= train_date_start) & (X['date'].dt.date <= train_date_end)][X.columns[:2]]
        Y_train = Y[(Y['date'].dt.date >= train_date_start) & (Y['date'].dt.date <= train_date_end)][Y.columns[:1]]
        X_test = X[(X['date'].dt.date >= test_date_start) & (X['date'].dt.date <= test_date_end)][X.columns[:2]]
        Y_test = Y[(Y['date'].dt.date >= test_date_start) & (Y['date'].dt.date <= test_date_end)][Y.columns[:1]]
        yield X_train, X_test, Y_train, Y_test

