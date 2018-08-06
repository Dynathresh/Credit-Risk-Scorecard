import os
from typing import Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
import math
import sklearn.linear_model as lm

def main():
    app_train = pd.read_csv('input/application_train.csv')
    features = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']
    target = 'TARGET'
    my_data = app_train[features]
    my_data = my_data.join(app_train[target])
    y = app_train[target]

    # .copy() -> deep copy the dataframe to preserve original data, just for insurance.
    # data_to_woe_numerical(my_data.copy(), target)
    data_woe_numerical_dict = data_to_woe_numerical(my_data.copy(), target)  # {characteristic: {(min,max):(woe,IV)}}

    data_replaced_with_woe = app_train[features].copy()  # Copy of dataframe, minus target column

    # Cast all feature values to float so the woe values that replace them remain as floats
    # Otherwise all the woe values get rounded to 0
    data_replaced_with_woe = data_replaced_with_woe.astype(np.float64)

    for feature in data_replaced_with_woe:
        woe_dict = data_woe_numerical_dict[feature]
        for i in data_replaced_with_woe.index:
            x = data_replaced_with_woe.at[i, feature]
            for key in woe_dict:
                if key[0] <= x < key[1]:
                    data_replaced_with_woe.at[i, feature] = woe_dict[key][0]  # Changes value into woe
                    break
    print(data_replaced_with_woe)


    X = data_replaced_with_woe
    lr = lm.LogisticRegression()
    lr.fit(X, y)
    print(lr.coef_)
    print(lr.intercept_)


def data_to_woe_numerical(data, target):
    #  instantiate WOE dictionary, with type hints
    woe_dict: Dict[Any, Dict[Tuple[Union[float, Any], Union[float, Any]], Tuple[float, Union[float, Any]]]] = {}
    print(type(data))
    for feature in data.columns[0:len(data.columns)-1]:
        single_feature_df = data.copy()
        single_feature_df = single_feature_df[[feature, target]]
        single_feature_df = single_feature_df.sort_values(by=[feature])
        value_counts = single_feature_df['TARGET'].value_counts()
        total_non_events = value_counts[0]  # Total zeroes
        total_events = value_counts[1]  # Total ones

        # TODO Find proper # of bins?
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html
        bins = np.array_split(single_feature_df, 10)  # Split into 10 equal bins
        null_bin = pd.DataFrame()

        bin_dict = {}
        bin_num = 0
        smallest_min = single_feature_df[feature].min()
        largest_max = single_feature_df[feature].max()
        for my_bin in bins:
            bin_num += 1

            # TODO Check if these will actually catch NULL/NaN/None values
            null_bin = pd.concat([null_bin, my_bin.loc[my_bin[feature].isnull()]], ignore_index=True)
            # null_bin.append(my_bin.loc[my_bin['AGE'].isnull()])  # Append is supposedly slower than concat  # TODO Null?
            my_bin = my_bin.drop(my_bin[my_bin[feature].isnull()].index)  # Removes null values from bin

            if my_bin[feature].min() != smallest_min:
                bin_min = my_bin[feature].min()  # Min value in bin
            else:
                bin_min = float('-inf')
            if my_bin[feature].max() != largest_max:
                bin_max = my_bin[feature].max()  # Max value in bin
            else:
                bin_max = float('inf')

            bin_value_counts = (my_bin['TARGET']).value_counts()
            non_events = (bin_value_counts[0])  # Non-Events in this bin
            events = (bin_value_counts[1])  # Events in this bin
            percent_non_events = non_events / total_non_events
            percent_events = events / total_events
            woe = math.log(percent_non_events / percent_events)
            IV = (percent_non_events - percent_events) * woe

            bin_dict[(bin_min, bin_max)] = (woe, IV)

        woe_dict[feature] = bin_dict

    # for key in woe_dict['DAYS_BIRTH']:
    #     print("{0}, {1}".format(key, woe_dict['DAYS_BIRTH'][key][0]))

    return woe_dict



# class RangeDict(dict):  # https://stackoverflow.com/a/39358140/9907619
#     def __getitem__(self, item):
#         if type(item) != range:
#             for key in self:
#                 return self[key]
#         else:
#             return super().__getitem__(item）



if __name__ == '__main__':
    main()
