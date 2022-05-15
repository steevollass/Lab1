import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
import os
from data_analysis import nans_count #, data_collector
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def delete_cols(df: pd.DataFrame, tresh=95):
    remove = []
    nans = nans_count(df)
    for feature in nans.keys():
        if nans[feature] > tresh:
            remove.append(feature)
    print('The sum of the rows we removing from the data: ', df[[c for c in df.columns if c in remove]].notna().sum())
    print('Number of features removed: ', len(remove))

    return df[[c for c in df.columns if c not in remove]]


def input_from_file(path, model):
    df = pd.read_csv(path, sep='|')
    id = str(path).split("_")[1].split(".")[0]
    df['id'] = len(df) * [int(id)]
    if 1 in list(df['SepsisLabel']):
        idx = list(df['SepsisLabel']).index(1)
        if model == 'baseline':
            df = df.iloc[idx]
        else:
            df = df[1:idx + 1]
    elif model == 'baseline':
        df = df.iloc[-1]
    return df[1:]


def get_input(path: str, model='Advanced'):

    input_df = pd.DataFrame()
    for file in tqdm(os.scandir(path)):
        file_input = input_from_file(file, model=model)
        input_df = input_df.append(file_input)

    return input_df


def clean_data(directory='train'):
    df = get_input(directory)
    nulls = nans_count(df)

    if directory == 'train':
        df = delete_cols(df, tresh=98)
    else:
        df = df[pd.read_csv('../Data/filled_train.csv').columns]
    descb = pd.read_csv('../Data/Statistical_stuff.csv', index_col='Unnamed: 0')
    tresh_null, tresh_std = 70, 7.5
    for col in tqdm(df.columns):
        for idx in df['id'].unique():
            patient = df[df['id'] == idx]

            if all(patient[col].isna()):
                if descb[col]['std'] < tresh_std:
                    patient[col].fillna(descb[col]['mean'])
                else:
                    patient[col].fillna(descb[col]['50%'])

            else:
                if 0 < nulls[col] < tresh_null:
                    patient[col].interpolate(limit_direction="both", inplace=True)
                else:
                    if descb[col]['std'] < tresh_std:
                        mean_non_missing = patient[col].mean()
                        patient[col].fillna(mean_non_missing, inplace=True)
                    else:
                        median_non_missing = patient[col].median()
                        patient[col].fillna(median_non_missing, inplace=True)
            df[df['id'] == idx] = patient
    df.to_csv('filled_' + directory + '.csv', index=False)
    return df


def main():
    clean_df = clean_data('train', 'light')



if __name__ == '__main__':
    main()
