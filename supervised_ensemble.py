import pandas as pd
from tqdm import tqdm
import pickle

from analysis import AnalysisTargets
from data import EquityData
from applied_stocks import applied_list


class EnsembleObjective(AnalysisTargets):
    applied_datasets = dict()
    filename = 'ensemble.pkl'

    def __init__(self):
        AnalysisTargets.__init__(self)

    @property
    def stocks(self):
        return applied_list

    def build(self):
        for stock in tqdm(self.stocks, desc='Building Datasets'):
            self.applied_datasets[stock] = self.create_target_dataset(stock)

        with open(self.filename, 'w') as handle:
            pickle.dump(self, handle)


if __name__ == '__main__':
    EnsembleObjective().build()

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
