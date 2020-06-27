import modin.pandas as pd
import numpy as np
from .metrics import custom_metric, random_predictions
from sklearn.model_selection import KFold, train_test_split
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


def validate_on_holdout(data: pd.DataFrame, model, test_size=0.2, y_name='response_att',
                        take_top_ratio=0.25):
    """
    :param data: данные
    :param model: ранжирующая модель с fit и predict
    :param test_size: доля данных для теста
    :param y_name: название столбца с таргетом
    :param take_top_ratio: на кого мы смотрим сверху
    :return: метрики по кросс-валидации
    """

    train, test = train_test_split(data, test_size=test_size, stratify=data['group'],
                                   shuffle=True)
    model.fit(train.drop(['group', y_name], axis=1), train[y_name], train['group'],
              eval_data=(test.drop(['group', y_name], axis=1), test[y_name], test['group']))

    train['uplift'] = model.predict(train.drop([y_name, 'group'], axis=1))
    test['uplift'] = model.predict(test.drop([y_name, 'group'], axis=1))

    train_score, test_score = custom_metric(train, take_top_ratio), custom_metric(test, take_top_ratio)
    train_random, test_random = random_predictions(train, 3, take_top_ratio), \
                                random_predictions(test, 3, take_top_ratio)
    scores = {'train': train_score, 'test': test_score, 'train_random':
        train_random, 'test_random': test_random}
    return scores, model


class StupidModel:
    def __init__(self, param=None, blend_type='multiply'):
        if param is None:
            param = {'n_jobs': -1, 'n_estimators': 10, 'eval_metric': ['ndcg', 'map'],
                     'objective': 'rank:ndcg', 'verbose': True}
        elif 'verbose' not in param:
            param['verbose'] = True

        self.test_model = xgboost.XGBRanker(**param)
        self.param = param
        self.control_model = xgboost.XGBRanker(**param)
        self.blend_type = blend_type

    def fit(self, data, y_train, group, eval_data=None):
        test_mask = group == 'test'
        control_mask = group == 'control'
        test = data[test_mask]
        control = data[control_mask]

        val_X, val_y, val_group = eval_data
        val_test_mask = val_group == 'test'
        val_control_mask = val_group == 'control'
        val_test = val_X[val_test_mask]
        val_control = val_X[val_control_mask]

        if self.param['verbose']:
            print('Обучаем модель на тестовой группе:')
        self.test_model.fit(test, y_train[test_mask], [test_mask.sum()],
                            eval_set=[(val_test, val_y[val_test_mask])],
                            eval_group=[[val_test_mask.sum()]], verbose = self.param['verbose'])
        
        if self.param['verbose']:
            print('\nОбучаем модель на контрольной группе:')
        
        self.control_model.fit(control, y_train[control_mask], [control_mask.sum()],
                               eval_set=[(val_control, val_y[val_control_mask])],
                               eval_group=[[val_control_mask.sum()]], verbose = self.param['verbose'])

    def predict(self, data):
        test_ranks = self.test_model.predict(data).reshape(-1, 1)
        control_ranks = self.control_model.predict(data).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_test = scaler.fit_transform(test_ranks)
        scaled_control = scaler.fit_transform(control_ranks)
        if self.blend_type == 'multiply':
            return scaled_test * (1 - scaled_control)
        elif self.blend_type == 'subtract':
            return scaled_test - scaled_control
        elif self.blend_type == 'divide':
            return scaled_test / scaled_control
        else:
            raise ValueError('Беда с типом!')
