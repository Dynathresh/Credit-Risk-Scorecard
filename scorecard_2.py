import os
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import math
# import statsmodels.api as sm

def main():
    print(os.listdir('input'))

    # 0-Repaid; 1-Defaulted
    app_train = pd.read_csv('input/application_train.csv')
    print('Training data shape: ', app_train.shape)

    print(type(app_train['TARGET']))
    print(type(app_train))
    chosen_feature = input('Choose feature:')

    # Feature-Target-Data-Frame
    ftdf = pd.DataFrame()  # Dataframe containing 2 columns: Feature and TARGET
    if chosen_feature == 'DAYS_BIRTH':  # Specifically for converting AGE(negative days) to normal people AGE(years)
        ftdf = convert_age(app_train[chosen_feature])
        ftdf['TARGET'] = app_train['TARGET']
        print('Age now inverted and converted to years:')
    else:
        ftdf['FEATURE'] = app_train[chosen_feature]
        ftdf['TARGET'] = app_train['TARGET']

    # Greater negative correlation == less likely to default as value (age) increases
    print('Pearson correlation coefficient of {0} with TARGET: {1}'.format(chosen_feature, ftdf['FEATURE'].corr(app_train['TARGET'])))
    calculate_woe(ftdf)


def convert_age(app_train_DAYS_BIRTH):
    app_train_DAYS_BIRTH = abs(app_train_DAYS_BIRTH)  # Division results in float by default
    age_dataframe = pd.DataFrame()
    age_dataframe['FEATURE'] = app_train_DAYS_BIRTH/365
    return age_dataframe


def calculate_woe(ftdf):
    ftdf = ftdf.sort_values(by=['FEATURE'])
    value_counts = ftdf['TARGET'].value_counts()
    # print(value_counts)
    total_non_events = value_counts[0]  # Total zeroes
    total_events = value_counts[1]  # Total ones

    # TODO Find proper # of bins?
    bins = np.array_split(ftdf, 10)  # Split into 10 equal bins
    null_bin = pd.DataFrame()

    woe_list = []  # Create list of dicts, then convert to DataFrame. Efficient because concat recreates data structure
    bin_num = 0
    for my_bin in bins:
        bin_num += 1

        # TODO Check if these will actually catch NULL/NaN/None values
        null_bin = pd.concat([null_bin, my_bin.loc[my_bin['FEATURE'].isnull()]], ignore_index=True)
        # null_bin.append(my_bin.loc[my_bin['AGE'].isnull()])  # Append is supposedly slower than concat
        my_bin = my_bin.drop(my_bin[my_bin['FEATURE'].isnull()].index)  # Removes null values from bin

        wd: Dict[str, Union[Union[str, float], Any]] = {}
        # print('Rows: ' + str(my_bin.shape[0]/app_train.shape[0]))  # % of all rows in this bin

        bin_min = my_bin['FEATURE'].min()  # Min bin
        bin_max = my_bin['FEATURE'].max()  # Max bin
        wd['RANGE'] = str(round(bin_min, 2)) + '-' + str(round(bin_max, 2))
        bin_value_counts = (my_bin['TARGET']).value_counts()
        wd['NON_EVENTS'] = (bin_value_counts[0])  # Non-Events in this bin
        wd['EVENTS'] = (bin_value_counts[1])  # Events in this bin
        percent_non_events = wd['NON_EVENTS'] / total_non_events
        percent_events = wd['EVENTS'] / total_events
        wd['PERCENT_NON_EVENTS'] = str(round(percent_non_events * 100, 2)) + '%'  # % of Non-Events
        wd['PERCENT_EVENTS'] = str(round(percent_events * 100, 2)) + '%'  # % of Events
        woe = math.log(percent_non_events / percent_events)
        IV = (percent_non_events - percent_events) * woe
        # Rounding WOE and IV to 4 places - Affect results?
        wd['WOE'] = round(woe, 4)
        wd['IV'] = round(IV, 4)
        woe_list.append(wd)

    woe_df = pd.DataFrame(woe_list,
                          columns=['RANGE', 'NON_EVENTS', 'EVENTS', 'PERCENT_NON_EVENTS', 'PERCENT_EVENTS', 'WOE',
                                   'IV'])
    pd.set_option('display.expand_frame_repr', False)  # Allows DF to be displayed in full
    woe_df.at['TOTAL', 'IV'] = woe_df['IV'].sum()
    print(woe_df.head(11))
    # TODO Resolve null bin
    # print(null_bin.head())


if __name__ == '__main__':
    main()
