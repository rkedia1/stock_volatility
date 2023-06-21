# @author Evan McFall


import pandas as pd


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


