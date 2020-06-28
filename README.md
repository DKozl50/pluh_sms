# pluh_sms
## Решение задачи хакатона "Bigtarget" команды Pluh.

Этот репозиторий содержит файлы, связанные с анализом датасета, предоставленного в рамках хакатона, а также модели, оптимизирующие uplift по тестовой и контрольной группам. Структура:
1. [experiments](https://github.com/DKozl50/pluh_sms/tree/master/experiments) - Папка с jupyter notebooks, в которых выполняется анализ датасета и подбор различных моделей.
2. [features](https://github.com/DKozl50/pluh_sms/tree/master/features) - некоторые функции для генерации признаков.
3. [scripts](https://github.com/DKozl50/pluh_sms/tree/master/scripts) - папка с используемымми функциями. Итоговые модели лежат в [pipeline](https://github.com/DKozl50/pluh_sms/blob/master/scripts/pipeline.py) - файл с нашими моделями. Финальная модель - BlendedModel, использующая блендинг нескольких обученных моделей.
4. [submissions](https://github.com/DKozl50/pluh_sms/blob/master/submissions) - файлы с предсказаниями различных моделей.

Кроме того, в ветке [web](https://github.com/DKozl50/pluh_sms/tree/web) находятся исходники сайта с полученными во время анализа инсайтами.
