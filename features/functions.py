import pandas as pd
import numpy as np

# Any group number is in diapason from 20 to 79
all_groups = [i for i in range(20, 80)]

# This function returns columns from 'columns' list, which are met in columns of "data"
def get_columns_list(data, columns):
    return data.columns[data.columns.isin(columns)]

def preprocessing(data : pd.DataFrame):
    # Name of a feature : (columns to collect from, method)
    # methods : mean, max, min, sum
    collect_features = {
        'cheque_count_12m_sum':
            (get_columns_list(data, ['cheque_count_12m_g{}'.format(i) for i in all_groups]), 'sum'),
        'children': (['children'], 'max'),
        'crazy_purchases_cheque_count_12m':
            (['crazy_purchases_cheque_count_12m'], 'max'),
        'k_var_disc_share_6m_max':
            (get_columns_list(data, ['k_var_disc_share_6m_g{}'.format(i) for i in all_groups]), 'max'),
        'k_var_sku_price_6m_max':
            (get_columns_list(data, ['k_var_sku_price_6m_g{}'.format(i) for i in all_groups]), 'max'),
        'sale_sum_12m_sum':
            (get_columns_list(data, ['sale_sum_6m_g{}'.format(i) for i in all_groups]), 'sum'),
    }

    # Add collections
    for key in collect_features.keys():
        method = collect_features[key][1]

        if method == 'mean':
            data.loc[:, key] = data[collect_features[key][0]].mean(axis=1)
        elif method == 'sum':
            data.loc[:, key] = data[collect_features[key][0]].sum(axis=1)
        elif method == 'max':
            data.loc[:, key] = data[collect_features[key][0]].max(axis=1)
        elif method == 'min':
            data.loc[:, key] = data[collect_features[key][0]].min(axis=1)

    # Additional observations
    add_features = ['cheques_per_child', 'sales_per_child']
    data['cheques_per_child'] = data['cheque_count_12m_sum'] / (data['children'].fillna(0) + 1)
    data['sales_per_child'] = data['sale_sum_12m_sum'] / (data['children'].fillna(0) + 1)

    # Separation parameter
    whis = 1.5

    data_map = np.array([])
    # Cycle by chosen features
    for key in list(collect_features.keys()) + add_features:

        # Drop outliers
        IQR = data[key].quantile(0.75) - data[key].quantile(0.25)

        if not data_map.size:
            data_map = (data[key] <= data[key].quantile(0.75) + IQR * whis).values
        else:
            data_map = data_map & (data[key] <= data[key].quantile(0.75) + IQR * whis).values

    data = data[data_map]

    index_columns = ['gender', 'main_format', 'children']

    # Let's fill NaN in "index_columns"
    data['group'].replace({'test': 1, 'control': 0}, inplace=True)
    data['gender'].replace({'М': 2, 'Ж': 1, 'Не определен': 0, None: 0}, inplace=True)
    data['children'].replace({None: -1}, inplace=True)

    # Group mean values
    group_means = data.groupby(index_columns).mean()
    group_means.fillna(group_means.mean(), inplace=True)

    for i in group_means.index:
        data.loc[(data['gender'] == i[0]).values &
                 (data['main_format'] == i[1]).values &
                 (data['children'] == i[2]).values] = data.loc[(data['gender'] == i[0]).values &
                                                               (data['main_format'] == i[1]).values &
                                                               (data['children'] == i[2]).values].fillna(
            group_means.loc[i])

def feature_generation(train : pd.DataFrame, test : pd.DataFrame):
    original_columns = train.columns

    for sum_feature in ['cheque_count_12m_g{}', 'cheque_count_3m_g{}',
                        'cheque_count_6m_g{}', 'sale_count_12m_g{}',
                        'sale_count_6m_g{}', 'sale_count_3m_g{}',
                        'sale_sum_6m_g{}', 'sale_sum_3m_g{}',
                        'sale_sum_12m_g{}'
                        ]:
        train.loc[:, sum_feature[:-4] + '_sum'] = train[
            get_columns_list(train, [sum_feature.format(i) for i in all_groups])].sum(axis=1)
        test.loc[:, sum_feature[:-4] + '_sum'] = test[
            get_columns_list(train, [sum_feature.format(i) for i in all_groups])].sum(axis=1)

    for mean_feature in ['k_var_count_per_cheque_1m_g{}',
                         'k_var_count_per_cheque_3m_g{}',
                         'k_var_count_per_cheque_6m_g{}',
                         'k_var_disc_share_1m_g{}',
                         'k_var_disc_share_3m_g{}',
                         'k_var_disc_share_6m_g{}',
                         'k_var_sku_price_3m_g{}',
                         'k_var_sku_price_6m_g{}',
                         'k_var_sku_price_1m_g{}',
                         ]:
        train.loc[:, sum_feature[:-4] + '_mean'] = train[
            get_columns_list(train, [mean_feature.format(i) for i in all_groups])].mean(axis=1)
        test.loc[:, sum_feature[:-4] + '_mean'] = train[
            get_columns_list(test, [mean_feature.format(i) for i in all_groups])].mean(axis=1)

    for over_feature in [('sale_sum_6m_sum', 'sale_count_6m_sum'),
                         ('sale_sum_12m_sum', 'sale_count_12m_sum'),
                         ('sale_sum_3m_sum', 'sale_count_3m_sum'),

                         ]:
        if (train[over_feature[1]] == 0).any():
            print("признак", over_feature, "не может быть создан: деление на 0")
            continue

        train[over_feature[0] + '/' + over_feature[1]] = \
            train[over_feature[0]] / train[over_feature[1]]
        test[over_feature[0] + '/' + over_feature[1]] = \
            test[over_feature[0]] / train[over_feature[1]]

    for prod_feature in [('mean_discount_depth_15d', 'sale_count_3m_sum'),
                         ('promo_share_15d', 'sale_count_3m_sum'),
                         ('k_var_sku_per_cheque_15d', 'promo_share_15d'),
                         ]:
        train[over_feature[0] + '*' + over_feature[1]] = \
            train[over_feature[0]] * train[over_feature[1]]
        test[over_feature[0] + '*' + over_feature[1]] = \
            test[over_feature[0]] * test[over_feature[1]]

'''
usage example :
from functions import preprocessing, feature_generation

preprocessing(train)
feature_generation(train, test)
'''