import pandas as pd
import numpy as np


def custom_metric(answers: pd.DataFrame, take_top_ratio=0.25):
    """
    Функция, возвращающая скор по ответам
    :param pd.DataFrame answers: датафрейм, в котором есть столбец uplift (скор модели), group - группа,
    response_att - отклик
    :param float take_top_ratio: доля людей, по которым мы считаем скор
    """
    answers.sort_values(by='uplift', inplace=True, ascending=False)
    n_samples = int(np.ceil(answers.shape[0] * take_top_ratio))
    answers = answers.iloc[:n_samples, :]
    answers_test = answers[answers['group'] == 'test']['response_att'].sum() / \
                   answers[answers['group'] == 'test'].shape[0]
    answers_control = answers[answers['group'] == 'control']['response_att'].sum() / \
                      answers[answers['group'] == 'control'].shape[0]
    return (answers_test - answers_control) * 100
