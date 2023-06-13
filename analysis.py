import pandas as pd

from ta.volatility import BollingerBands
from ta.trend import MACD


from data import EquityData


class AnalysisTargets(EquityData):
    base_datasets = dict()

    def __init__(self):
        EquityData.__init__(self)

    def create_target_dataset(self, equity: str,
                              deviation_window: int=2,
                              rolling_change_window: int=2):
        '''

        :param equity: the stock in question
        :param deviation_window: interval in days to observe the standard deviation of the intraday percentage change
                                 should be an indicator of the intraday volatility observed in trading patterns
        :param rolling_change_window: a string representing the period of time to look at price change;
                                      will default to 2 days
        :return:
        '''
        # creates a name to avoid having to pull the same data multiple times
        datasets = dict()
        intervals = ['1h', '1d']
        for interval in intervals:
            name = f'{equity}_{interval}'.lower()
            if name not in self.base_datasets:
                self.base_datasets[name] = self.get_data(equity, interval)

            try:
                df = self.base_datasets[name].copy()
                if interval == '1h':
                    datasets['Deviation'] = self.create_intraday_dataset(df, deviation_window)
                elif interval == '1d':
                    datasets['Base'] = self.base_datasets[name].copy()
            except Exception as e:
                print(e)

        if all(x in datasets for x in ['Deviation', 'Base']):
            applied_dataset = datasets['Base'].copy()
            applied_dataset['Rolling Change'] = applied_dataset['Close'].pct_change(rolling_change_window)
            applied_dataset = pd.merge(applied_dataset, datasets['Deviation'], left_index=True, right_index=True)
            print(applied_dataset)

    def create_intraday_dataset(self, df: pd.DataFrame,
                                deviation_window: int):
        df['pct_change'] = df['Close'].pct_change() * 100

        df.dropna(inplace=True)
        df.index = pd.DatetimeIndex(df.index)

        # in the case the below doesn't make sense
        # https://stackoverflow.com/questions/24875671/resample-in-a-rolling-window-using-pandas
        rolling_std = pd.DataFrame(df[['pct_change']].rolling(f'{deviation_window}d'). \
                                   std().resample('1d').first())
        rolling_std.columns = ['Deviation']
        return rolling_std


if __name__ == '__main__':
    # TODO sys.argv
    dataset = AnalysisTargets().create_target_dataset('PG')


 
# class TechnicalWrapper(object):
#     attributes = {'MACD': {'method': MACD,
#                            'inputs': ['window_slow', 'window_fast', 'window_sign'],
#                            'outputs': ['macd_signal', 'macd', 'macd_diff'],
#                            },
#                   }
#     def __init__(self):
#         pass
#
# class TechnicalAnalysis(EquityData):
#     def __init__(self):
#         EquityData.__init__(self)
#
#
#     def analyze(self):
