import pandas as pd
from tqdm import tqdm
import pickle
import os
import random

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
from helpers import err_handle, merge_dataframes, remove_duplicate_index


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

        with open(self.filename, 'wb') as handle:
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


class Model(EnsembleObjective):
    random_state = 42
    technical_datasets = dict()

    _sentiment_dataset = None

    def __init__(self):
        EnsembleObjective.__init__(self)

    def create_model_datasets(self, objective: str or None = None,
                              y_shift: int = -2):
        '''
        :param objective: if a technical indicator should be used it will pass into the loop
        :param y_shift: the number of intervals to shift x data forward
                        this is only used for creating the target variable
                        or a dummy dataset

                        it defaults to -2 as we would assume that 2 intervals forward would be
                        predicting of the x instance currently
        :return:
        '''
        datasets = dict()
        for equity, dataset in self.applied_datasets.items():
            if objective is not None:
                dataset = self.create_objective(dataset, objective)

            applied_features = [x for x in dataset.columns
                                if any(feature in x for feature in ['feature_', 'Close'])]
            dataset.dropna(inplace=True)
            # TODO deviation rolling dependent shift feature test

            X, Y = dataset[applied_features], dataset['Deviation'].shift(y_shift)
            for col in applied_features:
                X[col] = X[col].pct_change()

            X.dropna(inplace=True)
            Y.dropna(inplace=True)
            X = X.loc[X.index.isin(Y.index)]
            Y = Y.loc[Y.index.isin(X.index)]
            X.index = [f'{pd.Timestamp(x).strftime("%Y_%m_%d")}_{equity}' for x in X.index]
            Y.index = [f'{pd.Timestamp(x).strftime("%Y_%m_%d")}_{equity}' for x in Y.index]

            datasets[equity] = {'X': X, 'Y': Y}
        return datasets

    def create_objective(self, dataset: pd.DataFrame,
                         objective: str):
        if objective == 'bollinger':
            bb = BollingerBands(close=dataset['Close'])
            # TODO IMO
            dataset['feature_upper'] = bb.bollinger_hband()
            dataset['feature_lower'] = bb.bollinger_lband()
            dataset['feature_pband'] = bb.bollinger_pband()
            dataset['feature_wband'] = bb.bollinger_wband()

        return dataset

    @staticmethod
    def temporal_train_test_split(X: pd.DataFrame, Y: pd.DataFrame) -> tuple:
        # TODO include a validation split for LGBM or XGB
        split_index = X.index[int(len(X.index) * .7)]
        xtrain, xtest, ytrain, ytest = X.loc[X.index < split_index], X.loc[X.index >= split_index], \
            Y.loc[Y.index < split_index], Y.loc[Y.index >= split_index]
        return xtrain, xtest, ytrain, ytest

    # @staticmethod
    # def create_x_y_temporal(datasets: dict) -> tuple:
    #     # concatenate all of the dataframes from the dictionary into a single X, Y
    #
    #     return X, Y

    def create_twitter_datasets(self) -> dict:
        sentiment_datasets = dict()
        path = 'tweets-with-sentiment'
        for filename in os.listdir(path):
            try:
                filename = f'{path}\\{filename}'
                equity = filename.rsplit('\\')[-1].replace('.csv', '')
                df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
                df.index = [pd.Timestamp(x).replace(tzinfo=None) for x in df.index]
                df = pd.DataFrame(df['sentiment_score'].resample('1d').mean().ffill())
                sentiment_datasets[equity] = df
            except Exception as e:
                print(e)
        return sentiment_datasets

    def apply_sentiment_data(self, data: pd.DataFrame,
                             sentiment_data: dict):
        try:
            datasets = list()
            for equity, df in sentiment_data.items():
                df_copy = df.copy()
                try:
                    df_copy.index = [f'{pd.Timestamp(x).strftime("%Y_%m_%d")}_{equity}' for x in df_copy.index]
                    datasets.append(df_copy)
                except Exception as e:
                    print(err_handle(e, __file__))
            dataset = pd.concat(datasets)
            data = merge_dataframes([data, dataset])
            return data
        except Exception as e:
            print(err_handle(e, __file__))
            return data

        # for equity in sentiment_data:
        #     try:
        #         data[equity]['X'] = pd.merge(pd.DataFrame(data[equity]['X']),
        #                                          pd.DataFrame(sentiment_data[equity]),
        #                                          left_index=True, right_index=True)
        #     except:
        #         pass
        # return data

    def apply_model(self,
                     model: any,
                     X: pd.DataFrame,
                     Y: pd.DataFrame) -> pd.DataFrame:
        try:
            X = pd.DataFrame(X)
            Y = pd.DataFrame(Y)
            x_cols = X.columns
            y_cols = Y.columns
            dataset = merge_dataframes([X, Y]).dropna()
            X = pd.DataFrame(dataset[x_cols])
            Y = pd.DataFrame(dataset[y_cols])
            xtrain, xtest, ytrain, ytest = self.temporal_train_test_split(X, Y)
            xtrain = xtrain.loc[xtrain.index.isin(ytrain.index)]
            ytrain = ytrain.loc[ytrain.index.isin(xtrain.index)]
            xtest = xtest.loc[xtest.index.isin(ytest.index)]
            ytest = ytest.loc[ytest.index.isin(xtest.index)]

            model.fit(xtrain, ytrain)
            pred = pd.DataFrame(model.predict(xtest), index=xtest.index)
            return pred
        except Exception as e:
            print(err_handle(e, __file__))

    def create_X_Y(self, datasets: dict):
        X = pd.concat([dataset['X'] for equity, dataset in datasets.items()])
        Y = pd.concat([dataset['Y'] for equity, dataset in datasets.items()])
        return X, Y

    @staticmethod
    def evaluate_results(datasets: dict):
        '''
        Lazy way to rename each column from a prediction list
        and print out the score of each

        ! Requires a dict item for Actuals

        :param datasets:
        :return:
        '''

        for key, val in datasets.items():
            datasets[key].columns = [val]

        dataset = merge_dataframes(datasets.values())
        dataset.dropna(inplace=True)

        for col in dataset.columns:
            print(col)
            print(MAE(dataset[col], dataset['Actual']))

    @property
    def sentiment_dataset(self):
        if self._sentiment_dataset is None:
            self._sentiment_dataset = self.create_twitter_datasets()
        return self._sentiment_dataset

    def create_datasets(self, ensemble: any):
        # no technical objective -> baseline model results
        datasets = self.create_model_datasets()
        X, Y = self.create_X_Y(datasets)
        baseline_prediction = self.apply_model(ensemble, X.copy(), Y.copy())
        baseline_prediction = remove_duplicate_index(baseline_prediction)

        actuals = remove_duplicate_index(Y)

        dummy_datasets = self.create_model_datasets(y_shift=3)
        _, dummy_prediction = self.create_X_Y(dummy_datasets)

        bollinger_X = self.create_objective(X.copy(), 'bollinger')
        bollinger_prediction = self.apply_model(ensemble, bollinger_X, Y.copy())

        sentiment_dataset = self.apply_sentiment_data(X.copy(), self.sentiment_dataset)
        sentiment_dataset.dropna(inplace=True)
        sentiment_prediction = self.apply_model(ensemble, sentiment_dataset, Y)

        applied_dataset = self.apply_sentiment_data(bollinger_X.copy(), self.sentiment_dataset)
        applied_dataset.dropna(inplace=True)
        applied_prediction = self.apply_model(ensemble, applied_dataset, Y)

        dataset = merge_dataframes([dummy_prediction,
                                    sentiment_prediction,
                                    bollinger_prediction,
                                    applied_prediction,
                                    baseline_prediction,
                                    actuals])
        dataset.columns = ['Dummy', 'Sentiment',
                           'Technical',
                           'Sentiment Technical',
                           'Baseline', 'Actuals']
        dataset.dropna(inplace=True)
        return dataset

    def run_analysis(self):
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from xgboost import XGBRegressor

        models = {'linear': LinearRegression(),
                  'rf': RandomForestRegressor(),
                  'gb': GradientBoostingRegressor(),
                  'lgbm': LGBMRegressor(),
                  'xgb': XGBRegressor()}

        for name, model in models.items():
            try:
                dataset = self.create_datasets(model)
                print(name)
                for col in dataset.columns:
                    print(col)
                    print(MAE(dataset[col], dataset['Actuals']))
            except Exception as e:
                print(err_handle(e, __file__))



    # def run_analysis(self, sentiment_data: dict = None):
    #     # no technical objective -> baseline model results
    #     datasets = self.create_model_datasets()
    #     X, Y = self.create_X_Y(datasets)
    #     baseline = self.simple_model(X.copy(), Y.copy())
    #
    #     dummy_datasets = self.create_model_datasets(y_shift=3)
    #     _, dummy_prediction = self.create_X_Y(dummy_datasets)
    #
    #     actuals = baseline['ytest'].sort_index()
    #     actuals = remove_duplicate_index(actuals)
    #
    #     baseline_prediction = baseline['pred']
    #     baseline_prediction = remove_duplicate_index(baseline_prediction)
    #
    #     bollinger_X = self.create_objective(X.copy(), 'bollinger')
    #     bollinger_prediction = self.simple_model(bollinger_X, Y.copy())['pred']
    #
    #     sentiment_dataset = self.apply_sentiment_data(X.copy(), sentiment_data)
    #     sentiment_dataset.dropna(inplace=True)
    #     sentiment_prediction = self.simple_model(sentiment_dataset, Y)['pred']
    #
    #     applied_dataset = self.apply_sentiment_data(bollinger_X.copy(), sentiment_data)
    #     applied_dataset.dropna(inplace=True)
    #     applied_prediction = self.simple_model(applied_dataset, Y)['pred']
    #
    #     adf = merge_dataframes([dummy_prediction,
    #                             sentiment_prediction,
    #                             bollinger_prediction,
    #                             applied_prediction,
    #                             baseline_prediction,
    #                             actuals])
    #     adf.columns = ['Dummy', 'Sentiment', 'Bollinger', 'Applied', 'Baseline', 'Actuals']
    #     adf.dropna(inplace=True)
    #     for col in adf.columns:
    #         print(col)
    #         print(MAE(adf[col], adf['Actuals']))


        feature_only = self.create_objective()


        baseline_scores = self.evaluate_baseline(datasets)

        X, Y = self.create_datasets()
        baseline_data = self.simple_model(X, Y)

        baseline_pred = baseline_data['pred'].resample('1d').mean()

        # TODO
        X, Y = self.create_datasets(additional_data)
        advanced_data = self.simple_model(X, Y)

        advanced_pred = advanced_data['pred'].resample('1d').mean()

        target = Y.resample('1d').mean()
        dataset = merge_dataframes([baseline_pred, advanced_pred, target]).dropna()

        dataset.columns = ['Baseline', 'Sentiment', 'Deviation']
        for col in dataset.columns:
            print(col)
            print(MAE(dataset[col], dataset['Deviation']))


        #
        #
        # scores = {'MAE': MAE(ytest, pred), 'MSE': MSE(ytest, pred)}
        # baseline_scores = self.evaluate_baseline(technical_datasets)
        # improvement = self.determine_improvement(scores, baseline_scores)
        #
        # print(improvement)

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
    m = Model()
    m.run_analysis()

    # TechnicalModel().run_analysis()


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


    # def create_technical_datasets(self):
    #     technical_datasets = dict()
    #     for equity, dataset in self.applied_datasets.items():
    #         bb = BollingerBands(close=dataset['Close'])
    #         # TODO IMO
    #         dataset['bb_upper'] = bb.bollinger_hband()
    #         dataset['bb_lower'] = bb.bollinger_lband()
    #         dataset['bb_pband'] = bb.bollinger_pband()
    #         dataset['bb_wband'] = bb.bollinger_wband()
    #
    #         applied_features = [x for x in dataset.columns
    #                             if any(feature in x for feature in ['bb_', 'Close'])]
    #         dataset.dropna(inplace=True)
    #         # TODO deviation rolling dependent shift feature
    #         X, Y = dataset[applied_features].shift(-2), dataset['Deviation']
    #         for col in applied_features:
    #             X[col] = X[col].pct_change()
    #         X.dropna(inplace=True)
    #         Y.dropna(inplace=True)
    #         X = X.loc[X.index.isin(Y.index)]
    #         Y = Y.loc[Y.index.isin(X.index)]
    #
    #         technical_datasets[equity] = {'X': X, 'Y': Y}
    #     return technical_datasets

    # def create_datasets(self, additional_data: dict = None):
    #     data = self.create_technical_datasets()
    #
    #     if additional_data is not None:
    #         for equity in data:
    #             try:
    #                 data[equity]['X'] = pd.merge(pd.DataFrame(data[equity]['X']),
    #                                                  pd.DataFrame(additional_data[equity]),
    #                                                  left_index=True, right_index=True)
    #             except:
    #                 pass

        # X = list()
        # Y = list()
        # for equity in data:
        #     x_df = data[equity]['X']
        #     x_df['Equity'] = equity
        #     y_df = data[equity]['Y']
        #     X.append(x_df)
        #     Y.append(y_df)
        #
        # X = pd.concat(X)
        # Y = pd.concat(Y)
        # return X, Y
