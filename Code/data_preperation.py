import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from data_analysis import data_collector
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import RobustScaler

class PatientDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.num_of_hours = data.shape[1]  # TODO data.shape(1)?
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # assuming tensor (batch_size, hours, features), we want to get elements by each patient.
        indices = list(range(idx * self.num_of_hours, idx * self.num_of_hours + self.num_of_hours))
        patient = self.data[torch.arange(self.data.size(0)), indices]
        sample = {"Data": patient, "Labels": label}
        return sample


def dataframe_to_tensor(df: pd.DataFrame, name: str):

    patients = []
    for idx in tqdm(df['id'].unique()):
        patient = df[df['id'] == idx]
        patient_tensor = torch.tensor(patient.drop(columns=['SepsisLabel', 'id']).values.astype(np.float32))
        patients.append(patient_tensor)
    # since different patients stayed for varying times in ICU, we padded the data to get equal shapes for everyone.
    data_tensor = pad_sequence(patients, batch_first=True)
    torch.save(data_tensor, name + '_tensor.pt')


def main():
    rs = RobustScaler()
    df_train = pd.read_csv('../Data/filled_train.csv')
    df_train.iloc[:, df_train.columns != 'id'] = rs.fit_transform(df_train.iloc[:, df_train.columns != 'id'])
    
    train_labels = torch.tensor(df_train['SepsisLabel'])
    torch.save(train_labels, 'train_labels_tensor.pt')
    del train_labels
    
    dataframe_to_tensor(df_train, 'train')
    del df_train
    
    df_test = pd.read_csv('../Data/filled_test.csv')
    df_test.iloc[:, df_test.columns != 'id'] = rs.fit_transform(df_test.iloc[:, df_test.columns != 'id'])
    
    test_labels = torch.tensor(df_test['SepsisLabel'])
    torch.save(test_labels, 'test_labels_tensor.pt')
    del test_labels
    
    dataframe_to_tensor(df_test, 'test')
    del df_test



if __name__ == '__main__':
    main()



