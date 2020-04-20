import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data_learn.csv', header=0, na_values='?', delimiter=';')

print('Количество строк и столбцов:', data.shape)
print('Первые пять строк:\n', data.head())

# Получим некоторую сводную информацию по всей таблице. 
# По умолчанию будет выдана информация только для количественных признаков. 
# Это общее их количество (count), среднее значение (mean), стандартное отклонение (std), 
# минимальное (min), макcимальное (max) значения, медиана (50%) и значения нижнего (25%) 
# и верхнего (75%) квартилей:
print('Разбивка по числовым данным:\n',data.describe(), '\n')

# Добавил код получения целевого признака и др признаков надо сделать его уже в csv файле
data['target'] = np.random.randint(0, 2, data.shape[0])
data['Исполнительность/ дисциплинированность'] = np.random.randint(0, 11, data.shape[0])
data['Коммуникабельность'] = np.random.randint(0, 11, data.shape[0])
data['Стрессоустойчивость'] = np.random.randint(0, 11, data.shape[0])
data['Лидерство'] = np.random.randint(0, 11, data.shape[0])
data['Порядочность/ честность'] = np.random.randint(0, 11, data.shape[0])
data['Энергичность'] = np.random.randint(0, 11, data.shape[0])


# ГОТОВИМ ДАННЫЕ
# Алгоритмы машинного обучения из библиотеки scikit-learn не работают напрямую с категориальными признаками и данными, 
# в которых имеются пропущенные значения. Поэтому вначале подготовим наши данные

#Нам не нужны ФИО, дата рождения выкинем их
X = data.drop('ФИО', axis='columns') 
X = X.drop('Дата рождения', axis='columns') 
X = X.drop('ID', axis='columns') 
X = X.drop('Семейное положение', axis='columns') 
X = X.drop('Отдел', axis='columns') 
X = X.drop('Должность', axis='columns') 
X = X.drop('Наличие правонарушений/ судимостей', axis='columns') 
X = X.drop('Трудовые заслуги', axis='columns') 
X = X.drop('Образование', axis='columns') 
#X = X.drop('Дополнительная квалификация', axis='columns') 
#Посмотрим заполнение признаков
print(X.count(axis=0))


# Категориальные признаки
categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']
print('Категориальные признаки:',categorical_columns, '\n')

# Числовые признаки
numerical_columns   = [c for c in X.columns if X[c].dtype.name != 'object']
print('Числовые признаки:',numerical_columns)

# Приведем категорилаьные признаки к числовому варианту
# Вначале выделим бинарные и небинарные признаки
X_describe = X.describe(include=[object])
binary_columns    = [c for c in categorical_columns if X_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if X_describe[c]['unique'] > 2]
print ('Бинарные признаки: ',binary_columns, '\n', 'Не бинарные признаки: ',nonbinary_columns)

#Преобразуем 'Дополнительная квалификация' в числ столбец если она есть то 1 нету 0
X.at[X['Дополнительная квалификация'] == 'Нет', 'Дополнительная квалификация'] = 0
X.at[X['Дополнительная квалификация'] != 0, 'Дополнительная квалификация'] = 1
print (X['Дополнительная квалификация'].head())

#Преобразуем 'Наличие выговоров по трудовой дисциплине' в числ столбец если она есть то 1 нету 0
X.at[X['Наличие выговоров по трудовой дисциплине'] == 'Нет', 'Наличие выговоров по трудовой дисциплине'] = 0
X.at[X['Наличие выговоров по трудовой дисциплине'] != 0, 'Наличие выговоров по трудовой дисциплине'] = 1
print (X['Наличие выговоров по трудовой дисциплине'].head())


#X_nonbinary = pd.get_dummies(X[nonbinary_columns])
#print (X_nonbinary.columns)

# Соединим обработанные данные вместе
#X = pd.concat((X[numerical_columns], X[binary_columns], X_nonbinary), axis=1)
#X = pd.DataFrame(X, dtype=float)


#Разделим данные на целевой признак и все отсальные данные
X = X.drop('target', axis='columns') 
y = data['target']

#Делим данные на обучающую выборку и тестовую 70 на 30 процентов
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 

#kNN – метод ближайших соседей
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)
print (err_train, err_test)

# Метод дерева решений
from sklearn import tree

model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)

model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)
print (err_train, err_test)

# Случайный лес
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print (err_train, err_test)

