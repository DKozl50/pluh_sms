import modin.pandas as pd
import numpy as np
from .metrics import custom_metric, random_predictions
from sklearn.model_selection import KFold
import xgboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score


def run(data: pd.DataFrame, model, n_splits=4, y_name='response_att', take_top_ratio=0.25):
    """
    :param data: данные
    :param model: ранжирующая модель с fit и predict
    :param n_splits: число разбиений для кросс-валидации
    :param y_name: название столбца с таргетом
    :param take_top_ratio: на кого мы смотрим сверху
    :return: метрики по кросс-валидации
    """
    kf = KFold(n_splits=n_splits)
    scores = {'train': [], 'test': [], 'train_random': [], 'test_random': []}
    for train_idx, test_idx in kf.split(data):
        y_train = data[y_name].loc[train_idx]
        y_test = data[y_name].loc[test_idx]

        train = data.loc[train_idx]
        test = data.loc[test_idx]

        model.fit(train.drop(y_name, axis=1), y_train)
        train['uplift'] = model.predict(train.drop([y_name, 'group'], axis=1))
        test['uplift'] = model.predict(test.drop([y_name, 'group'], axis=1))
        train_score, test_score = custom_metric(train, take_top_ratio), custom_metric(test, take_top_ratio)
        train_random, test_random = random_predictions(train, 3, take_top_ratio), \
                                    random_predictions(test, 3, take_top_ratio)
        scores['train'].append(train_score)
        scores['test'].append(test_score)

        scores['train_random'].append(train_random)
        scores['test_random'].append(test_random)

    means = {name: sum(scores[name]) / len(scores[name]) for name in scores.keys()}
    return means, scores


class StupidModel:
    def __init__(self, verbose=False):
        self.test_model = xgboost.XGBRanker(n_jobs=-1, n_estimators=10)
        self.control_model = xgboost.XGBRanker(n_jobs=-1, n_estimators=10)
        self.verbose = verbose

    def fit(self, data, y_train):
        test_mask = data['group'] == 'test'
        control_mask = data['group'] == 'control'
        test = data[test_mask].drop('group', axis=1)
        control = data[control_mask].drop('group', axis=1)
        self.test_model.fit(test, y_train[test_mask], [test_mask.sum()], verbose=self.verbose,
                           eval_set=[(test, y_train[test_mask])], eval_group=[[test_mask.sum()]], eval_metric='ndcg')
        self.control_model.fit(control, y_train[control_mask], [control_mask.sum()], verbose=self.verbose,
                           eval_set=[(control, y_train[control_mask])], eval_group=[[control_mask.sum()]], eval_metric='ndcg')

    def predict(self, data):
        test_ranks = self.test_model.predict(data).reshape(-1, 1)
        control_ranks = self.control_model.predict(data).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_test = scaler.fit_transform(test_ranks)
        scaled_control = scaler.fit_transform(control_ranks)
        return scaled_test - scaled_control
