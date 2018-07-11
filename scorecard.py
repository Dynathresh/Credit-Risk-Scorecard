import numpy as np
import pandas as pd
import os

print(os.listdir("input/"))

# 0-Repaid; 1-Defaulted
app_train = pd.read_csv('input/application_train.csv')
print('Training data shape: ', app_train.shape)

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])  # Division results in float by default
age_df = pd.DataFrame()
age_df['AGE'] = app_train['DAYS_BIRTH']/365
# Greater negative correlation == less likely to default as value (age) increases
print(age_df['AGE'].corr(app_train['TARGET']))

age_df = pd.concat([age_df, app_train['TARGET']], axis=1)
# print(age_df.head())

age_df = age_df.sort_values(by=['AGE'])
value_counts = age_df['TARGET'].value_counts()
# print(value_counts)
age_zeroes = value_counts[0]  # total zeroes
age_ones = value_counts[1]  # total ones

age_bins = np.array_split(age_df, 10)
null_bin = pd.DataFrame()

for bin in age_bins:
    null_bin.append(bin.loc[bin['AGE'].isnull()])  # Not sure if this will catch NULL values ##PLEASE CHECK##
    bin_min = bin['AGE'].min()
    bin_max = bin['AGE'].max()
    bin_value_counts = bin['TARGET'].value_counts()
    bin_zeroes = bin_value_counts[0]
    bin_ones = bin_value_counts[1]
    print(bin.shape)


def getWOE(bin):
    return