import modin.pandas as pd
import numpy as np
from metrics import custom_metric, random_predictions
from sklearn.model_selection import KFold, train_test_split
import xgboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


def importances(train, model):
    cols = list(train.columns)

    feats = pd.Series(data=model.feature_importances_, index=cols) 
    feats.sort_values(ascending=False, inplace=True)
    return feats[feats != 0]
    

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
                        take_top_ratio=0.25, verbose=False):
    """
    :param data: данные
    :param model: ранжирующая модель с fit и predict
    :param test_size: доля данных для теста
    :param y_name: название столбца с таргетом
    :param take_top_ratio: на кого мы смотрим сверху
    :return: метрики по кросс-валидации
    """

    train, test = train_test_split(data, test_size=test_size, stratify=data['group'],
                                   shuffle=True, random_state=42)
    model.fit(train, test)

    train['uplift'] = model.predict(train.drop([y_name, 'group'], axis=1))
    test['uplift'], logs = model.predict(test.drop([y_name, 'group'], axis=1), verbose=True)

    train_score, test_score = custom_metric(train, take_top_ratio), custom_metric(test, take_top_ratio)
    scores = {'train': train_score, 'test': test_score}
    if not verbose:
        return scores, model
    else:
        return scores, model, logs, test['uplift'], test


class StupidModel:
    def __init__(self, param=None, blend_type='multiply', scale=True, test_model=None, control_model=None,
                ranking=True):
        if param is None:
            param = {'n_jobs': -1, 'n_estimators': 10, 'eval_metric': ['ndcg', 'map'],
                     'objective': 'rank:ndcg', 'verbose': True}
        elif 'verbose' not in param:
            param['verbose'] = True

        self.test_model = test_model
        self.control_model = control_model
            
        if self.test_model is None:
            self.test_model = xgboost.XGBRanker(**param)
        self.param = param
        if self.control_model is None:
            self.control_model = xgboost.XGBRanker(**param)
        self.blend_type = blend_type
        self.scale = scale
        self.ranking = ranking

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
        if self.ranking:
            self.test_model.fit(test, y_train[test_mask], [test_mask.sum()],
                                eval_set=[(val_test, val_y[val_test_mask])],
                                eval_group=[[val_test_mask.sum()]], verbose = self.param['verbose'])
        else:
            self.test_model.fit(test, y_train[test_mask],
                                eval_set=[(val_test, val_y[val_test_mask])],
                                verbose = self.param['verbose'])
        
        if self.param['verbose']:
            print('\nОбучаем модель на контрольной группе:')
        
        if self.ranking:
            self.control_model.fit(control, y_train[control_mask], [control_mask.sum()],
                                   eval_set=[(val_control, val_y[val_control_mask])],
                                   eval_group=[[val_control_mask.sum()]], verbose = self.param['verbose'])
        else:
            self.control_model.fit(control, y_train[control_mask],
                                   eval_set=[(val_control, val_y[val_control_mask])],
                                   verbose = self.param['verbose'])

    def predict(self, data, verbose=False):
        if self.ranking:
            test_ranks = self.test_model.predict(data)
            control_ranks = self.control_model.predict(data)
        else:
            test_ranks = self.test_model.predict_proba(data)[:,1]
            control_ranks = self.control_model.predict_proba(data)[:,1]
        scaler = MinMaxScaler()
        if self.scale:
            scaled_test = scaler.fit_transform(test_ranks.reshape(-1, 1))
            scaled_control = scaler.fit_transform(control_ranks.reshape(-1, 1))
        else:
            scaled_test = test_ranks
            scaled_control = control_ranks
        
        if self.blend_type == 'multiply':
            res = scaled_test * (1 - scaled_control)
        elif self.blend_type == 'subtract':
            res = scaled_test - scaled_control
        elif self.blend_type == 'divide':
            res = scaled_test / scaled_control
        else:
            raise ValueError('Беда с типом!')
        
        if verbose:
            return res, scaled_test, scaled_control
        else:
            return res


class IndianModel:
    def __init__(self, param=None, model=None):
        if param is None:
            param = {'n_jobs': -1, 'n_estimators': 10, 'eval_metric': ['ndcg', 'map'],
                     'objective': 'rank:ndcg', 'verbose': True}
        elif 'verbose' not in param:
            param['verbose'] = True

        self.model = model

        if self.model is None:
            self.model = xgboost.XGBClassifier(**param)
        self.param = param

    def fit(self, data, eval_data=None):
        """
        В дате должны быть столбцы group и response_att
        """

        data['class'] = 0
        data.loc[(data['group'] == 'control') & (data['response_att'] == 1), 'class'] = 1
        data.loc[(data['group'] == 'test') & (data['response_att'] == 0), 'class'] = 2
        data.loc[(data['group'] == 'test') & (data['response_att'] == 1), 'class'] = 3

        eval_data['class'] = 0
        eval_data.loc[(eval_data['group'] == 'control') & (eval_data['response_att'] == 1), 'class'] = 1
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 0), 'class'] = 2
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 1), 'class'] = 3

        y_train, y_test = data['class'].values, eval_data['class'].values
        X_train = data.drop(['class', 'group', 'response_att'], axis=1).values
        X_test = eval_data.drop(['class', 'group', 'response_att'], axis=1).values
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        data.drop('class', axis=1, inplace=True)
        eval_data.drop('class', axis=1, inplace=True)

    def predict(self, data, verbose=False):
        if type(data) == pd.DataFrame:
            assert {'class', 'response_att', 'group'}.intersection(set(data.columns)) == set(), "Oops!"
            data = data.values
            print('here')
        try:
            data = data.values
        except:
            pass
        pred = self.model.predict_proba(data)
        final = pred[:, 3] + pred[:, 0] - pred[:, 1] - pred[:, 2]
        if verbose:
            return final, pred
        else:
            return final


class StackedModel:
    def __init__(self, models=None):
        self.models = models

    def fit(self, data, eval_data=None):
        """
        В дате должны быть столбцы group и response_att
        """

        data['class'] = 0
        data.loc[(data['group'] == 'control') & (data['response_att'] == 1), 'class'] = 1
        data.loc[(data['group'] == 'test') & (data['response_att'] == 0), 'class'] = 2
        data.loc[(data['group'] == 'test') & (data['response_att'] == 1), 'class'] = 3

        eval_data['class'] = 0
        eval_data.loc[(eval_data['group'] == 'control') & (eval_data['response_att'] == 1), 'class'] = 1
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 0), 'class'] = 2
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 1), 'class'] = 3

        y_train, y_test = data['class'].values, eval_data['class'].values
        X_train = data.drop(['class', 'group', 'response_att'], axis=1).values
        X_test = eval_data.drop(['class', 'group', 'response_att'], axis=1).values
        for model in self.models:
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            print(str(model)[:20] + f'\nТочность: {acc}\n ROC AUC: {auc}')
        data.drop('class', axis=1, inplace=True)
        eval_data.drop('class', axis=1, inplace=True)

    def predict(self, data, verbose=False):
        if type(data) == pd.DataFrame:
            assert {'class', 'response_att', 'group'}.intersection(set(data.columns)) == set(), "Oops!"
            data = data.values
            print('here')
        try:
            data = data.values
        except:
            pass
        
        finals = []
        for model in self.models:
            pred = model.predict_proba(data)
            finals.append(pred[:, 3] + pred[:, 0] - pred[:, 1] - pred[:, 2])
        final = sum(finals) / len(finals)
        if verbose:
            return final, finals
        else:
            return final
        

class SemiStackedModel:
    def __init__(self, models, top_model):
        self.models = models
        self.top_model = top_model

    def fit(self, data, eval_data=None):
        """
        В дате должны быть столбцы group и response_att
        """

        data['class'] = 0
        data.loc[(data['group'] == 'control') & (data['response_att'] == 1), 'class'] = 1
        data.loc[(data['group'] == 'test') & (data['response_att'] == 0), 'class'] = 2
        data.loc[(data['group'] == 'test') & (data['response_att'] == 1), 'class'] = 3

        eval_data['class'] = 0
        eval_data.loc[(eval_data['group'] == 'control') & (eval_data['response_att'] == 1), 'class'] = 1
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 0), 'class'] = 2
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 1), 'class'] = 3

        y_train, y_test = data['class'].values, eval_data['class'].values
        X_train = data.drop(['class', 'group', 'response_att'], axis=1).values
        X_test = eval_data.drop(['class', 'group', 'response_att'], axis=1).values
        top_feats_train = []
        top_feats_test = []
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            print(name + f'\nТочность: {acc}\nROC AUC: {auc}')
            top_feats_train.append(model.predict_proba(X_train))
            top_feats_test.append(model.predict_proba(X_test))
            
        top_mat_train = np.hstack(top_feats_train)
        top_mat_test = np.hstack(top_feats_test)
        
        
        self.top_model.fit(top_mat_train, y_train)
        acc = accuracy_score(y_test, self.top_model.predict(top_mat_test))
        auc = roc_auc_score(y_test, self.top_model.predict_proba(top_mat_test), multi_class='ovr')
        print(f'\nМодель верхнего уровня \nТочность: {acc}\nROC AUC: {auc}')
        
        data.drop('class', axis=1, inplace=True)
        eval_data.drop('class', axis=1, inplace=True)

    def predict(self, data, verbose=False):
        if type(data) == pd.DataFrame:
            assert {'class', 'response_att', 'group'}.intersection(set(data.columns)) == set(), "Oops!"
            data = data.values
            print('here')
        try:
            data = data.values
        except:
            pass
        
        preds = []
        for name, model in self.models.items():
            pred = model.predict_proba(data)
            preds.append(pred)
        top_matrix = np.array(preds).reshape(-1, len(self.models) * 4)
        pred = self.top_model.predict_proba(top_matrix)
        
        final = pred[:, 3] + pred[:, 0] - pred[:, 1] - pred[:, 2]
        
        if verbose:
            return final, pred
        else:
            return final        
        
        
class BlendedModel:
    def __init__(self, models, verbose=True):
        self.models = models
        self.verbose = verbose

    def fit(self, data, eval_data=None):
        """
        В дате должны быть столбцы group и response_att
        """

        data['class'] = 0
        data.loc[(data['group'] == 'control') & (data['response_att'] == 1), 'class'] = 1
        data.loc[(data['group'] == 'test') & (data['response_att'] == 0), 'class'] = 2
        data.loc[(data['group'] == 'test') & (data['response_att'] == 1), 'class'] = 3

        eval_data['class'] = 0
        eval_data.loc[(eval_data['group'] == 'control') & (eval_data['response_att'] == 1), 'class'] = 1
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 0), 'class'] = 2
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 1), 'class'] = 3

        y_train, y_test = data['class'].values, eval_data['class'].values
        X_train = data.drop(['class', 'group', 'response_att'], axis=1).values
        X_test = eval_data.drop(['class', 'group', 'response_att'], axis=1).values
        top_feats_train = []
        top_feats_test = []
        for name, model in self.models.items():
            if self.verbose:
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
            else:
                model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            print('\n' + name + f'\nТочность: {acc}\nROC AUC: {auc}')
            top_feats_train.append(model.predict_proba(X_train))
            top_feats_test.append(model.predict_proba(X_test))
            
        train_proba = sum(top_feats_train) / len(top_feats_train)
        test_proba = sum(top_feats_test) / len(top_feats_test)       
        
        #acc = accuracy_score(y_test, test_proba)
        auc = roc_auc_score(y_test, test_proba, multi_class='ovr')
        print(f'\nМодель верхнего уровня \nROC AUC: {auc}')
        
        data.drop('class', axis=1, inplace=True)
        eval_data.drop('class', axis=1, inplace=True)

    def predict(self, data, verbose=False):
        if type(data) == pd.DataFrame:
            assert {'class', 'response_att', 'group'}.intersection(set(data.columns)) == set(), "Oops!"
            data = data.values
            print('here')
        try:
            data = data.values
        except:
            pass
        
        preds = []
        for name, model in self.models.items():
            pred = model.predict_proba(data)
            preds.append(pred)
        
        pred = sum(preds) / len(preds)
        
        final = pred[:, 3] + pred[:, 0] - pred[:, 1] - pred[:, 2]
        
        if verbose:
            return final, pred
        else:
            return final        
        
        
def night_preprocess(data):
    
    data['gender'].replace({'Ж': 0, 'М': 1, 'Не определен': np.NaN}, inplace=True)
    cols_now = list(data.columns)
    
    
    data['response_std'] = data[['response_sms', 'response_viber']].std(axis=1)
    data['response_multi'] = data['response_sms'] * data['response_viber']

    cols = ['days_between_visits_15d', 'discount_depth_15d', 'discount_depth_1m']
    for col in cols:
        data['mean_' + col] = data['stdev_' + col] / data['k_var_' + col]
        
    months = ['3m', '6m', '12m']
    for s in months:
        for i in range(10, 60):
            count = 'sale_count_' + s + '_g' + str(i)
            sums = 'sale_sum_' + s + '_g' + str(i)
            if (count in data.columns) and (sums in data.columns):
                data['sale_mean_' + s + '_g' + str(i)] = data[sums] / data[count]
                print(sums, count)

    data['use_of_perdelta'] = data['perdelta_days_between_visits_15_30d'] * data['mean_days_between_visits_15d']
    
    data['mean_crazy_6m'] = data['crazy_purchases_goods_count_6m'] / data['crazy_purchases_cheque_count_6m']
    data['mean_crazy_12m'] = data['crazy_purchases_goods_count_12m'] / data['crazy_purchases_cheque_count_12m']
    
    data['food_growth'] = data['food_share_15d'] / data['food_share_1m']
    
    new_cols = list(set(data.columns).difference(cols_now))
    return new_cols


class ExperimentalModel:
    def __init__(self, models, verbose=True):
        self.models = models
        self.verbose = verbose

    def fit(self, data, eval_data=None):
        """
        В дате должны быть столбцы group и response_att
        """

        data.loc[(data['group'] == 'control') & (data['response_att'] == 1), 'class'] = 1
        data.loc[(data['group'] == 'control') & (data['response_att'] == 0), 'class'] = 0
        data.loc[(data['group'] == 'test') & (data['response_att'] == 0), 'class'] = 1
        data.loc[(data['group'] == 'test') & (data['response_att'] == 1), 'class'] = 0

        eval_data.loc[(eval_data['group'] == 'control') & (eval_data['response_att'] == 1), 'class'] = 1
        eval_data.loc[(eval_data['group'] == 'control') & (eval_data['response_att'] == 0), 'class'] = 0
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 0), 'class'] = 1
        eval_data.loc[(eval_data['group'] == 'test') & (eval_data['response_att'] == 1), 'class'] = 0

        y_train, y_test = data['class'].values, eval_data['class'].values
        X_train = data.drop(['class', 'group', 'response_att'], axis=1).values
        X_test = eval_data.drop(['class', 'group', 'response_att'], axis=1).values
        top_feats_train = []
        top_feats_test = []
        for name, model in self.models.items():
            if self.verbose:
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
            else:
                model.fit(X_train, y_train)
            #acc = accuracy_score(y_test, model.predict(X_test))
            #auc = roc_auc_score(y_test, model.predict_proba(X_test))
            #print('\n' + name + f'\nТочность: {acc}\nROC AUC: {auc}')
            top_feats_train.append(model.predict_proba(X_train))
            top_feats_test.append(model.predict_proba(X_test))
            
        train_proba = sum(top_feats_train) / len(top_feats_train)
        test_proba = sum(top_feats_test) / len(top_feats_test)       
                                                      
        
        #acc = accuracy_score(y_test, test_proba)
        #auc = roc_auc_score(y_test, test_proba)
        #print(f'\nМодель верхнего уровня \nROC AUC: {auc}')
        
        data.drop('class', axis=1, inplace=True)
        eval_data.drop('class', axis=1, inplace=True)

    def predict(self, data, verbose=False):
        if type(data) == pd.DataFrame:
            assert {'class', 'response_att', 'group'}.intersection(set(data.columns)) == set(), "Oops!"
            data = data.values
            print('here')
        try:
            data = data.values
        except:
            pass
        
        preds = []
        for name, model in self.models.items():
            pred = model.predict_proba(data)
            preds.append(pred)
        
        pred = sum(preds) / len(preds)
        
        final = pred[:,0] - pred[:,1]
        
        if verbose:
            return final, pred
        else:
            return final        
        
        