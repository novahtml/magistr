import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_learn.csv', header=0, na_values='?', delimiter=';')

print('Количество строк и столбцов:', data.shape)
print('Первые пять строк:\n', data.head())

#Получим некоторую сводную информацию по всей таблице. 
#По умолчанию будет выдана информация только для количественных признаков. 
#Это общее их количество (count), среднее значение (mean), стандартное отклонение (std), 
#минимальное (min), макcимальное (max) значения, медиана (50%) и значения нижнего (25%) 
#и верхнего (75%) квартилей:
print(data.describe(), '\n')

#Категориальные признаки
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
print('Категориальные признаки:',categorical_columns, '\n')

#Числовые признаки
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print('Числовые признаки:',numerical_columns)