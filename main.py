import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

data = pd.read_csv('data_learn.csv', header=0, na_values='?', delimiter=';')

print('Количество строк и столбцов:', data.shape)
print('Первые пять строк:\n', data.head())

# Получим некоторую сводную информацию по всей таблице. 
# По умолчанию будет выдана информация только для количественных признаков. 
# Это общее их количество (count), среднее значение (mean), стандартное отклонение (std), 
# минимальное (min), макcимальное (max) значения, медиана (50%) и значения нижнего (25%) 
# и верхнего (75%) квартилей:
print('Разбивка по числовым данным:\n',data.describe(), '\n')

# Диаграмма расеивания для признаков 'Рабочий стаж (лет)' и 'Наличие правонарушений/ судимостей'
col1 = 'Рабочий стаж (лет)'
col2 = 'Наличие правонарушений/ судимостей'

plt.figure(figsize=(10, 6))

plt.scatter(data[col1][data['Кандидат на увольнение'] == 1],
            data[col2][data['Кандидат на увольнение'] == 1],
            alpha=0.75,
            color='red',
            label='1')

plt.scatter(data[col1][data['Кандидат на увольнение'] == 0],
            data[col2][data['Кандидат на увольнение'] == 0],
            alpha=0.75,
            color='blue',
            label='0')

plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(loc='best');

# ГОТОВИМ ДАННЫЕ
# Алгоритмы машинного обучения из библиотеки scikit-learn не работают напрямую с категориальными признаками и данными, 
# в которых имеются пропущенные значения. Поэтому вначале подготовим наши данные

#Убираем не нужые столбцы
X = data.drop('ФИО', axis='columns') 
X = X.drop('Дата рождения', axis='columns') 
X = X.drop('ID', axis='columns') 
X = X.drop('Семейное положение', axis='columns') 
X = X.drop('Отдел', axis='columns') 
X = X.drop('Должность', axis='columns') 
X = X.drop('Дети', axis='columns') 
X = X.drop('Образование', axis='columns') 

#Посмотрим заполнение признаков
print(X.count(axis=0))

# Категориальные признаки
categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']
print('Категориальные признаки:',categorical_columns, '\n')

# Числовые признаки
numerical_columns   = [c for c in X.columns if X[c].dtype.name != 'object']
print('Числовые признаки: ',numerical_columns)

# Приведем категорилаьные признаки к числовому варианту
X[categorical_columns].describe()
# Категориалные признаки бинарного типа, приведем их к числовому варианту

#Преобразуем 'Дополнительная квалификация' в числ столбец если она есть то 1 нету 0
X.at[X['Дополнительная квалификация'] == 'Нет', 'Дополнительная квалификация'] = 0
X.at[X['Дополнительная квалификация'] != 0, 'Дополнительная квалификация'] = 1


#Преобразуем 'Наличие выговоров по трудовой дисциплине' в числ столбец если она есть то 1 нету 0
X.at[X['Наличие выговоров по трудовой дисциплине'] == 'Нет', 'Наличие выговоров по трудовой дисциплине'] = 0
X.at[X['Наличие выговоров по трудовой дисциплине'] != 0, 'Наличие выговоров по трудовой дисциплине'] = 1


#Преобразуем 'Наличие правонарушений/ судимостей' в числ столбец если она есть то 1 нету 0
X.at[X['Наличие правонарушений/ судимостей'] == 'Нет', 'Наличие правонарушений/ судимостей'] = 0
X.at[X['Наличие правонарушений/ судимостей'] != 0, 'Наличие правонарушений/ судимостей'] = 1


#Преобразуем 'Наличие правонарушений/ судимостей' в числ столбец если она есть то 1 нету 0
X.at[X['Трудовые заслуги'] == 'Нет', 'Трудовые заслуги'] = 0
X.at[X['Трудовые заслуги'] != 0, 'Трудовые заслуги'] = 1

#Из построенных диаграмм видно, что признаки не сильно коррелируют между собой, 
#что впрочем можно также легко установить, посмотрев на корреляционную матрицу. 
#Все ее недиагональные значения по модулю не превосходят 0.4:
matrix = X.copy()
matrix.columns = ['Col' + str(i) for i in range(1, 13)] 

print(matrix.corr())

# Выделим целевой признак отдельно
X = X.drop('Кандидат на увольнение', axis='columns') 
y = data['Кандидат на увольнение']

#Делим данные на обучающую выборку и тестовую 70 на 30 процентов
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 

#kNN – метод ближайших соседей
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print('Точность прогноза на тестовом наборе: ', knn.score(X_test,y_test))

# SVC – машина опорных векторов- любит нормализации количественных признаков
from sklearn.svm import SVC
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)

print('Точность прогноза на тестовом наборе: ', svc.score(X_test,y_test))


# Метод дерева решений
from sklearn import tree

tree_model = tree.DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train, y_train)

print('Точность прогноза на тренировочном наборе: ', tree_model.score(X_test,y_test))

import graphviz 
from sklearn.tree import export_graphviz
dot_data = tree.export_graphviz(tree_model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("tree_model")

tree.plot_tree(tree_model.fit(X, y)) 
dot_data = tree.export_graphviz(tree_model, out_file=None, 
filled=True, rounded=True,  
 special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# Случайный лес
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(X_train, y_train)

print('Точность прогноза на тренировочном наборе: ', rf.score(X_test,y_test))



