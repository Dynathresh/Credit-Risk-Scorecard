import numpy as np
import pandas as pd
import os

print(os.listdir("input/"))

# 0-Repaid; 1-Defaulted
app_train = pd.read_csv('input/application_train.csv')
print('Training data shape: ', app_train.shape)

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])  # Division results in float by default
col_age_years = app_train['DAYS_BIRTH']/365
# Greater negative correlation == less likely to default as value (age) increases
print(col_age_years.corr(app_train['TARGET']))

age_df = pd.concat([col_age_years, app_train['TARGET']], axis=1)
print(age_df.head())

age_df = age_df.sort_values(by=['DAYS_BIRTH'])
print(age_df.head(20))

bins_age = pd.cut(age_df['DAYS_BIRTH'], 3)
print(bins_age)
print(age_df.shape)
