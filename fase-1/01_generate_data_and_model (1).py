# -*- coding: utf-8 -*-
"""01_generate_data_and_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/ibitabadger/Models1/blob/main/01_generate_data_and_model.ipynb
"""

!pip install lightgbm==3.2.1

import pandas as pd
import gdown
# Se define un diccionario con los tipos de datos específicos para cada columna de un dataframe de pandas.
train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
# Se define un diccionario de números atómicos para elementos químicos
ATOMIC_NUMBERS = {
    'H': 1,  #Hidrógeno
    'C': 6,  #Carbono
    'N': 7,  #Nitrógeno
    'O': 8,  #Oxígeno
    'F': 9   #Fluor
}
#Mostrar cadenas de texto largas sin truncarlas
pd.set_option('display.max_colwidth', None)
#mostrar hasta 120 filas antes de truncar la salida
pd.set_option('display.max_rows', 120)
#mostrar hasta 120 columnas antes de truncar la salida
pd.set_option('display.max_columns', 120)
#Dataframe magnetic shielding tensors
driveUrlMagnetic = 'https://drive.google.com/uc?id=1sP_bFjh0UuC1hMtAvDzAXetzwp698yJs'
outputFileMagnetic = 'magnetic_shielding_tensors.csv'
gdown.download(driveUrlMagnetic, outputFileMagnetic, quiet=False)
df_train_sub_tensor = pd.read_csv(outputFileMagnetic, delimiter = ';')
print(df_train_sub_tensor.head())

#Dataframe mulliken charges
driveUrlMulliken = 'https://drive.google.com/uc?id=1ERHKxkbCSTSxFXRXVB5uzd8TgHGkSjjq'
outputFileMulliken = 'mulliken_charges.csv'
gdown.download(driveUrlMulliken, outputFileMulliken, quiet=False)
df_train_sub_charge = pd.read_csv(outputFileMulliken, delimiter = ';')
print(df_train_sub_charge.head())

#Dataframe structures
structures_dtypes = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}
driveUrlStructures = 'https://drive.google.com/uc?id=10un5w-BDlBxgi2WIgmLDGINwSfoPU1t2'
outputFileStructures = 'structures.csv'
gdown.download(driveUrlStructures, outputFileStructures, quiet=False)
structures_csv = pd.read_csv(outputFileStructures, dtype=structures_dtypes)
structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')
structures_csv.head(10)

# Imprimir la forma del DataFrame `structures_csv`, mostrando el número de filas y columnas.
print('Shape: ', structures_csv.shape)

# Calcular y mostrar la cantidad total de memoria (en bytes) que el DataFrame `structures_csv` está utilizando.
print('Total: ', structures_csv.memory_usage().sum())

# Ejecutar `memory_usage()` en `structures_csv` para mostrar el uso de memoria de cada columna individualmente.
structures_csv.memory_usage()

#Dataframe test. Conjunto de prueba
driveUrlTest = 'https://drive.google.com/uc?id=1U7X0SLBAEFbNaDa2CHtOBHkYA9iTzGmm'
outputFileTest = 'test.csv'
gdown.download(driveUrlTest, outputFileTest, quiet=False)
test_csv = pd.read_csv(outputFileTest, index_col='id', dtype=train_dtypes)

#Añadir columna 'molecule_index'. Identificador numérico para cada molécula
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')

#Selección del subconjunto de columnas para conservar en el DataFrame `test_csv`.
test_csv = test_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]
test_csv.head(10)

#Dataframe sample.

driveUrlSample = 'https://drive.google.com/uc?id=1lMSEThXn_JmEerxPZQLHCDwotj3zpe4A'
outputFileSample = 'test.csv'
gdown.download(driveUrlSample, outputFileSample, quiet=False)
submission_csv = pd.read_csv(outputFileSample, index_col='id')
print(submission_csv.head())

#Dataframe train

driveUrlTrain = 'https://drive.google.com/uc?id=1cFjgpNoWLIFtvfBKqsxeZ_vWNeuelhwh'
outputFileTrain = 'train.csv'
gdown.download(driveUrlTrain, outputFileTrain, quiet=False)
train_csv = pd.read_csv(outputFileTrain, index_col='id', dtype=train_dtypes)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]
train_csv.head(10)

print('Shape: ', train_csv.shape)
print('Total: ', train_csv.memory_usage().sum())
train_csv.memory_usage()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import numpy as np

import math
import gc
import copy

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor

#Función para filtrar por tipo de acopamiento dado

def build_type_dataframes(base, structures, coupling_type):
    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
    base = base.reset_index()
    base['id'] = base['id'].astype('int32')
    structures = structures[structures['molecule_index'].isin(base['molecule_index'])]
    return base, structures

#Función para tomar un conjunto base de datos (base), combinarlo con un conjunto de estructuras moleculares (structures) para añadir las coordenadas (x, y, z) de los átomos específicos
def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df

#Función para fusionar los DataFrames base y atoms en función de las columnas 'molecule_index', 'atom_index_0', y 'atom_index_1'
def add_atoms(base, atoms):
    df = pd.merge(base, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    return df

#Función para combinar los DataFrames base y structures  y luego filtrar las filas donde los índices de átomos coinciden
def merge_all_atoms(base, structures):
    df = pd.merge(base, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'])
    df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]
    return df

#Función para calcular el centro entre dos puntos de coordenadas espaciales de átomos
def add_center(df):
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))

#Función para calcular la distancia euclidiana entre cada punto representado por las coordenadas (x, y, z)

def add_distance_to_center(df):
    df['d_c'] = ((
        (df['x_c'] - df['x'])**np.float32(2) +
        (df['y_c'] - df['y'])**np.float32(2) +
        (df['z_c'] - df['z'])**np.float32(2)
    )**np.float32(0.5))

#Función para calcular la distancia euclidiana entre dos puntos especificados por los sufijos de columnas dados (suffix1 y suffix2)
def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) +
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))

#Función para calcular todas las distancias entre pares de átomos en el DataFrame
def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])

    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)

#Función para agregar información sobre el número de átomos de cada molécula al DataFrame base
def add_n_atoms(base, structures):
    dfs = structures['molecule_index'].value_counts().rename('n_atoms').to_frame()
    return pd.merge(base, dfs, left_on='molecule_index', right_index=True)

#Función para construir un dataframe con características específicas de tipo de acoplamiento químico
def build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=10):
    base, structures = build_type_dataframes(some_csv, structures_csv, coupling_type)
    base = add_coordinates(base, structures, 0)
    base = add_coordinates(base, structures, 1)

    base = base.drop(['atom_0', 'atom_1'], axis=1)
    atoms = base.drop('id', axis=1).copy()
    if 'scalar_coupling_constant' in some_csv:
        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)

    add_center(atoms)
    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

    atoms = merge_all_atoms(atoms, structures)
    add_distance_to_center(atoms)

    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
    atoms['num'] = atom_groups.cumcount() + 2
    atoms = atoms.drop(['d_c'], axis=1)
    atoms = atoms[atoms['num'] < n_atoms]

    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    atoms = atoms.reset_index()

    # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            atoms[col] = atoms[col].fillna(0).astype('int8')

    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')
    full = add_atoms(base, atoms)
    add_distances(full)

    full.sort_values('id', inplace=True)

    return full

#Función para tomar un DataFrame df y un número máximo de átomos n_atoms para construir un nuevo DataFrame con las características seleccionadas.
def take_n_atoms(df, n_atoms, four_start=4):
    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)

    for i in range(n_atoms):
        num = min(i, 4) if i < four_start else 4
        for j in range(num):
            labels.append(f'd_{i}_{j}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    return df[labels]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# full = build_couple_dataframe(train_csv, structures_csv, '1JHN', n_atoms=10)
# print(full.shape)

full.columns

df = take_n_atoms(full, 7)
# LightGBM performs better with 0-s then with NaN-s
df = df.fillna(0)
df.columns

X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
y_data = df['scalar_coupling_constant'].values.astype('float32')

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=128)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}

#Entrenamiento del modelo de regresión con el algoritmo LGBMRegressor
model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
        verbose=True, early_stopping_rounds=200)

y_pred = model.predict(X_val)
np.log(mean_absolute_error(y_val, y_pred))

#Importancia de cada característica den el modelo entrenado
cols = list(df.columns)
cols.remove('scalar_coupling_constant')
cols
df_importance = pd.DataFrame({'feature': cols, 'importance': model.feature_importances_})
sns.barplot(x="importance", y="feature", data=df_importance.sort_values('importance', ascending=False));

def build_x_y_data(some_csv, coupling_type, n_atoms):
    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)

    df = take_n_atoms(full, n_atoms)
    df = df.fillna(0)
    print(df.columns)

    if 'scalar_coupling_constant' in df:
        X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
        y_data = df['scalar_coupling_constant'].values.astype('float32')
    else:
        X_data = df.values.astype('float32')
        y_data = None

    return X_data, y_data

#Entrenar  un modelo de regresión para un tipo específico de acoplamiento químico y realizar predicciones en el conjunto de prueba
def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms, n_folds=5, n_splits=5, random_state=128):
    print(f'*** Training Model for {coupling_type} ***')

    X_data, y_data = build_x_y_data(train_csv, coupling_type, n_atoms)
    X_test, _ = build_x_y_data(test_csv, coupling_type, n_atoms)
    y_pred = np.zeros(X_test.shape[0], dtype='float32')

    cv_score = 0

    if n_folds > n_splits:
        n_splits = n_folds

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        if fold >= n_folds:
            break
        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)
        model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
            verbose=100, early_stopping_rounds=200)

        y_val_pred = model.predict(X_val)
        val_score = np.log(mean_absolute_error(y_val, y_val_pred))
        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')

        cv_score += val_score / n_folds
        y_pred += model.predict(X_test) / n_folds


    submission.loc[test_csv['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred
    return cv_score

model_params = {
    '1JHN': 7,
    '1JHC': 10,
    '2JHH': 9,
    '2JHN': 9,
    '2JHC': 9,
    '3JHH': 9,
    '3JHC': 10,
    '3JHN': 10
}
N_FOLDS = 3
submission = submission_csv.copy()

cv_scores = {}
for coupling_type in model_params.keys():
    cv_score = train_and_predict_for_one_coupling_type(
        coupling_type, submission, n_atoms=model_params[coupling_type], n_folds=N_FOLDS)
    cv_scores[coupling_type] = cv_score

#Puntajes de validación cruzada para diferentes tipos de acoplamiento químico
pd.DataFrame({'type': list(cv_scores.keys()), 'cv_score': list(cv_scores.values())})

#Promedio de los puntajes de validación cruzada
np.mean(list(cv_scores.values()))

submission[submission['scalar_coupling_constant'] == 0].shape

submission.head(10)