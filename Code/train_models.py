import pandas as pd
import os
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
import pickle
from feature_enginner import get_input

import warnings
warnings.filterwarnings("ignore")


def get_filled_input(subset: str, model: str):
    df = pd.read_csv('../Data/filled_' + subset + '.csv')
    input_df = pd.DataFrame()

    for idx in df['id'].unique():
        patient = df[df['id'] == idx]
        if model == 'baseline':
            input_df = input_df.append(patient.iloc[-1])

        else:
            input_df = input_df.append(patient)

    return input_df


def flatten_input(df: pd.DataFrame):
    patient_list = []
    for idx in df['id'].unique():
        patient = df[df['id'] == idx].drop(columns='id').to_numpy().flatten()

        patient_list.append(patient)

    return patient_list


def baseline_MLP(data_type='basic'):
    print("\nMLP:")
    if data_type == 'basic':
        layers = 100

        train_data = get_input('../Data/train', model='baseline').fillna(0)
        test_data = get_input('../Data/test', model='baseline').fillna(0)
    else:
        layers = (100, 200, 100)
        train_data = get_filled_input('train', model='baseline')
        test_data = get_filled_input('test', model='baseline')

    activation = 'tanh'
    alpha = 0.001
    lr = 0.001

    clf = MLPClassifier(hidden_layer_sizes=layers, random_state=1, max_iter=300,
                        n_iter_no_change=10, activation=activation, learning_rate_init=lr,
                        alpha=alpha).fit(train_data.drop(['SepsisLabel','id'], axis=1),
                                                 train_data['SepsisLabel'])
    y_pred = clf.predict(test_data.drop(['SepsisLabel', 'id'], axis=1))
    pickle.dump(clf, open('../Models/Baseline' + data_type + '.sav', 'wb'))
    f1 = f1_score(test_data['SepsisLabel'], np.array(y_pred))
    print(f'\nF1 score on test is: ', f1)


def flattened_data_MLP(data_type='basic'):
    if data_type == 'basic':
        train_data = get_input('../Data/train', model='flattened')
        test_data = get_input('../Data/test', model='flattened')
    else:
        train_data = get_filled_input('train', model='flattened')
        test_data = get_filled_input('train', model='flattened')

    train_labels = train_data['SepsisLabel']
    train_data = flatten_input(train_data.drop(columns='SepsisLabel'))

    test_labels = test_data['SepsisLabel']
    test_data = flatten_input(test_data.drop(columns='SepsisLabel'))

    layers = (100, 200, 100)
    clf = MLPClassifier(hidden_layer_sizes=layers, random_state=1, max_iter=300).fit(train_data, train_labels)
    y_pred = clf.predict(test_data)
    pickle.dump(clf, open('../Models/Advanced' + data_type + '.sav', 'wb'))
    f1 = f1_score(test_labels, np.array(y_pred))
    print(f'\nF1 score on test is: ', f1)


def main():
    # baseline_MLP()
    flattened_data_MLP()



if __name__ == '__main__':
    main()
