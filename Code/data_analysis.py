import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def data_collector(number_of_files=20000, phase='train'):

    df = pd.read_csv("../Data/" + phase + "/patient_0.psv", sep='|')
    df['id'] = 0  # TODO beware of test id
    for i in tqdm(range(1, number_of_files)):
        tmp = pd.read_csv("../Data/" + phase + "/patient_" + str(i) + ".psv", sep='|')
        tmp['id'] = i
        df = pd.concat([df, tmp])

    return df


def nans_count(df: pd.DataFrame):
    percent_missing = df.isnull().sum() * 100 / len(df)
    nans = {k: v for k, v in sorted(percent_missing.to_dict().items(), key=lambda item: item[1], reverse=True)}

    return nans


def histograms(df: pd.DataFrame, presets: list):
    # histograms of different features
    for (col, xlab, color) in presets:
        hist = plt.figure()
        plt.hist(df[col], color=color)
        plt.title('Histogram of:' + xlab)
        plt.xlabel(xlab)
        plt.ylabel('Count in data')
        hist.savefig('../Plots/Histogram of: ' + col)  # TODO fuck your mom (somehow make it work)


def correlation(df: pd.DataFrame):
    # TODO make it normal to use
    corrs = {}
    for col in df:
        corrs[col] = df.corr()[col].abs().sort_values(ascending=False)[1:6]


    # heatmap = sns.heatmap(df.corr(), annot=True, cmap='BrBG')
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
    # plt.savefig('correlation.png', bbox_inches='tight')


def feature_distribution(data: pd.DataFrame):
    # Each row is now a patient (not an hour)
    df_per_person = data.copy().drop_duplicates(subset=['id'], keep='last').astype({'Age': 'int'})
    descb = data.describe()
    if not os.path.isfile('../Data/Statistical_stuff.csv'):
        descb.to_csv('../Data/Statistical_stuff.csv')

    # Sick vs Not sick
    # plot0 = plt.figure()
    # plt.hist(x=df_per_person['SepsisLabel'], bins=2, orientation='vertical')
    # plot0.show()
    num_sick = df_per_person.groupby('SepsisLabel').size()
    plot0 = plt.figure()
    plt.barh(['Healthy', 'Sick'], num_sick.values)
    plt.title('Illness prevalence')
    plt.xlabel('Sick or not')
    plt.ylabel('Count')
    if os.path.isfile('../Plots/figure0.png'):
        os.remove('../Plots/figure0.png')
    plot0.savefig('../Plots/figure0')
    # count data by gender
    males_df = df_per_person[df_per_person['Gender'] == 1]
    females_df = df_per_person[df_per_person['Gender'] == 0]

    diff_gender = len(males_df/len(females_df))
    print("The difference between males and females is", diff_gender, "\n")

    #Ages distb
    bins = [17, 30, 40, 50, 60, 70, 80, 90, 120]  # TODO minimum in age?
    labels = ['17-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    df_per_person['Agerange'] = pd.cut(df_per_person['Age'], bins, labels=labels, include_lowest=True)
    ages_sum = df_per_person.groupby('Agerange').size()
    plot1 = plt.figure()
    plt.bar(labels, ages_sum.tolist())
    plt.title('Distribution of ages in data')
    plt.xlabel('Age groups')
    plt.ylabel('Count')
    if os.path.isfile('../Plots/figure1.png'):
        os.remove('../Plots/figure1.png')
    plot1.savefig('../Plots/figure1')

    # Chance of illness by age and gender
    plot2 = plt.figure()
    males_df['Agerange'] = pd.cut(males_df['Age'], bins, labels=labels, include_lowest=True)
    females_df['Agerange'] = pd.cut(females_df['Age'], bins, labels=labels, include_lowest=True)
    males_sick_mean = males_df.groupby(['Agerange'])['SepsisLabel'].mean()
    females_sick_mean = females_df.groupby(['Agerange'])['SepsisLabel'].mean()
    plt.plot(sorted(males_df['Agerange'].unique()), males_sick_mean, label='Male')
    plt.plot(sorted(females_df['Agerange'].unique()), females_sick_mean, label='Females')
    plt.title('Chance for sickness for gender and age')
    plt.xlabel('Age range')
    plt.ylabel('Illness Chance')
    plt.legend()
    if os.path.isfile('../Plots/figure2.png'):
        os.remove('../Plots/figure2.png')
    plot2.savefig('../Plots/figure2')


    # histograms
    presets = [('HospAdmTime', 'Hospital Admission Times', 'blue'),
               ('ICULOS', 'ICU Admission Times', 'red'),
               ('HR', 'Heart Beats Per Minute', 'seagreen')]  # TODO check Hosp distb on the whole data
    histograms(df_per_person, presets)

    correlation(data)
    # [[c for c in data.columns if c in
    #                       ['HR', 'Age', 'Temp', 'O2Sat', 'ICULOS', 'HospAdmTime', 'SepsisLabel']]]

    # TODO diff between sick genders


def ratios(number_of_files, phase):
    df = pd.read_csv("../Data/" + phase + "/patient_0.psv", sep='|').tail(1)
    df['id'] = 0  # TODO beware of test id
    for i in tqdm(range(1, number_of_files)):
        tmp = pd.read_csv("../Data/" + phase + "/patient_" + str(i) + ".psv", sep='|').tail(1)
        tmp['id'] = i
        df = pd.concat([df, tmp])

    men_df = df[df.Gender == 1]
    women_df = df[df.Gender == 0]

    print(f'{len(df[df.SepsisLabel == 1])/len(df)} of the patients are sick')
    print(f'{len(men_df)/len(df)} of the patients are male')
    print(f'{len(men_df[men_df.SepsisLabel == 1])/len(men_df)} of the male patients were sick')
    print(f'{len(women_df[women_df.SepsisLabel == 1])/len(women_df)} of the female patients were sick')


def dora_the_data_explorer():

    # head = pd.read_csv("../Data/train/patient_0.psv", sep='|')
    # print("Those are features in our dataset:", list(head.columns))
    # print("Number of features:", len(list(head.columns)))

    df = data_collector(20000, phase='train')
    # ratios(20000, 'train')
    # nans = nans_count(df)
    feature_distribution(df)
    # print(nans)


def main():
    dora_the_data_explorer()

if __name__ == '__main__':
    main()