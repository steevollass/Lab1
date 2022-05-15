import sys
from feature_enginner import get_input
import pickle
import numpy as np
from sklearn.metrics import f1_score


def main():
    directory = sys.argv[1]
    print(f'Loading patients from {directory}')
    test_data = get_input(directory, model='baseline').fillna(0)
    loaded_model = pickle.load(open('../Models/Baselinebasic.sav', 'rb'))

    print(f'Performing prediction...')
    y_pred = loaded_model.predict(test_data.drop(['SepsisLabel', 'id'], axis=1))
    f1 = f1_score(test_data['SepsisLabel'], np.array(y_pred))
    print(f'\nF1 score on test is: ', f1)


if __name__ == '__main__':
    main()
