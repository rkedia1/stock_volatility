import pandas as pd
from tqdm import tqdm
import pickle
import os

# Feature methods
from ta.volatility import BollingerBands
from ta.trend import MACD

# Models and SK
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error as MAE, \
    mean_absolute_percentage_error as MAPE, \
    mean_squared_error as MSE

# Local methods
from analysis import AnalysisTargets
from data import EquityData
from applied_stocks import applied_list


class EnsembleObjective(AnalysisTargets):
    applied_datasets = dict()
    filename = 'applied_datasets.pkl'

    def __init__(self):
        AnalysisTargets.__init__(self)
        if not os.path.isfile(self.filename):
            self.build()
        else:
            with open(self.filename, 'rb') as handle:
                self.applied_datasets = pickle.load(handle)

    @property
    def stocks(self):
        return applied_list

    def build(self):
        for stock in tqdm(self.stocks, desc='Building Datasets'):
            self.applied_datasets[stock] = self.create_target_dataset(stock)

        with open('ensemble.pkl', 'wb') as handle:
            pickle.dump(self.applied_datasets, handle)


class VisualObjective(EnsembleObjective):
    _dataset = pd.DataFrame()
    def __init__(self):
        EnsembleObjective.__init__(self)

    @property
    def data(self):
        if self._dataset.empty:
            for stock in self.stocks:
                self._dataset = pd.concat([self._dataset, self.applied_datasets[stock]])
        return self._dataset

    def plot(self):
        dataset = self.data.copy()
        print(dataset)

class TechnicalModel(EnsembleObjective):
    random_state = 42
    technical_datasets = dict()

    def __init__(self):
        EnsembleObjective.__init__(self)

    def create_technical_datasets(self):
        technical_datasets = dict()
        for equity, dataset in self.applied_datasets.items():
            bb = BollingerBands(close=dataset['Close'])
            # TODO IMO
            dataset['bb_upper'] = bb.bollinger_hband()
            dataset['bb_lower'] = bb.bollinger_lband()
            dataset['bb_pband'] = bb.bollinger_pband()
            dataset['bb_wband'] = bb.bollinger_wband()

            applied_features = [x for x in dataset.columns
                                if any(feature in x for feature in ['bb_', 'Close'])]
            dataset.dropna(inplace=True)
            # TODO deviation rolling dependent shift feature
            X, Y = dataset[applied_features], dataset['Deviation'].shift(-2)
            for col in applied_features:
                X[col] = X[col].pct_change()
            X.dropna(inplace=True)
            Y.dropna(inplace=True)
            X = X.loc[X.index.isin(Y.index)]
            Y = Y.loc[Y.index.isin(X.index)]

            technical_datasets[equity] = {'X': X, 'Y': Y}
        return technical_datasets

    @staticmethod
    def temporal_train_test_split(X: pd.DataFrame, Y: pd.DataFrame) -> tuple:
        # TODO include a validation split for LGBM or XGB
        split_index = X.index[int(len(X.index) * .7)]
        xtrain, xtest, ytrain, ytest = X.loc[X.index < split_index], X.loc[X.index >= split_index], \
            Y.loc[Y.index < split_index], Y.loc[Y.index >= split_index]
        return xtrain, xtest, ytrain, ytest

    def run_analysis(self, additional_data: None):
        # TODO pass params down
        technical_datasets = self.create_technical_datasets()
        # TODO if additional data is passed; merge here to show benefit;
        # will need to know method to merge data as this is a dictionary by equity

        # concatenate all of the dataframes from the dictionary into a single X, Y
        X, Y = pd.DataFrame(), pd.DataFrame()
        X = pd.concat([technical_datasets[equity]['X'] for equity in technical_datasets])
        Y = pd.concat([technical_datasets[equity]['Y'] for equity in technical_datasets])
        # build a train test instance
        xtrain, xtest, ytrain, ytest = self.temporal_train_test_split(X, Y)

        model = LGBMRegressor(random_state=self.random_state)
        model.fit(xtrain, ytrain)
        pred = pd.DataFrame(model.predict(xtest), index=xtest.index)

        scores = {'MAE': MAE(ytest, pred), 'MSE': MSE(ytest, pred)}
        baseline_scores = self.evaluate_baseline(technical_datasets)
        improvement = self.determine_improvement(scores, baseline_scores)

    @staticmethod
    def determine_improvement(scoreset_a: dict, scoreset_b: dict) -> dict:
        improvement = dict()
        for key, val in scoreset_a.items():
            try:
                improvement[key] = scoreset_b[key] - scoreset_a[key]
            except: pass
        return improvement

    def evaluate_baseline(self, technical_datasets: dict) -> dict:
        baseline = {'MAE': list(),
                    'MSE': list()}
        for equity in technical_datasets:
            try:
                subset_y = technical_datasets[equity]['Y']

                # TO DENOTE; this is missing two samples but is nominal in comparison
                # TODO set n instance looking forward
                baseline_assumption = subset_y.shift(2)
                baseline_dataset = pd.merge(baseline_assumption, subset_y, left_index=True, right_index=True).dropna()
                baseline_dataset.columns = ['Pred', 'Actual']

                subset_mae = MAE(baseline_dataset['Pred'], baseline_dataset['Actual'])
                subset_mse = MSE(baseline_dataset['Pred'], baseline_dataset['Actual'])
                baseline['MAE'].append(subset_mae)
                baseline['MSE'].append(subset_mse)
            except Exception as e:
                print(e)

        baseline_scores = {key : sum(vals) / len(vals) for key, vals in baseline.items()}
        return baseline_scores



if __name__ == '__main__':
    TechnicalModel().create_technical_datasets()


#
#
#
#             from lightgbm import LGBMRegressor
#             model = LGBMRegressor()
#
#             # TODO include a validation split for LGBM or XGB
#             split_index = X.index[int(len(X.index) * .7)]
#             xtrain, xtest, ytrain, ytest = X.loc[X.index < split_index], X.loc[X.index >= split_index],\
#                                            Y.loc[Y.index < split_index], Y.loc[Y.index >= split_index]
#
#             model.fit(xtrain, ytrain)
#
#             pred = pd.DataFrame(model.predict(xtest), index=xtest.index)
#
#             score = MAE(pred, ytest)
#
#             dummy_pred = ytest.shift(2)
#             ndf = pd.merge(ytest, dummy_pred, left_index=True, right_index=True).dropna()
#             ndf.columns = ['Dummy Pred', 'Actual']
#
#             dummy_score = MAE(ndf['Dummy Pred'], ndf['Actual'])
#
#
#
# # class TechnicalWrapper(object):
# #     attributes = {'MACD': {'method': MACD,
# #                            'inputs': ['window_slow', 'window_fast', 'window_sign'],
# #                            'outputs': ['macd_signal', 'macd', 'macd_diff'],
# #                            },
# #                   }
# #     def __init__(self):
# #         pass
# #
# # class TechnicalAnalysis(EquityData):
# #     def __init__(self):
# #         EquityData.__init__(self)
# #
# #
# #     def analyze(self):
