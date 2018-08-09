import os
from typing import Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
import math
import sklearn.linear_model as lm

# output现在会存到一个文件里

def main():
    app_train = pd.read_csv('input/application_train.csv')  # 把training data变成dataframe
    # TODO algorithmically选最好的特徵
    feature_list = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']  # 特征list
    target = 'TARGET'  # 目标柱名字
    train_data = app_train[feature_list]  # 把想用的数据提取到自己的dataframe
    train_data = train_data.join(app_train[target])  # 把目标柱加到dataframe里

    # .copy() -> 深拷贝dataframe所以不会不小心把原数据改变了
    # 格式：{特征： {(minimum threshold, maximum threshold):(woe,IV)}}
    data_woe_numerical_dict = get_woe_numerical_dict(train_data.copy(), target)

    # 把数据的原始数字变成woe
    data_replaced_with_woe = replace_with_woe(app_train[feature_list].copy(), data_woe_numerical_dict)

    X = data_replaced_with_woe
    y = app_train[target]  # 用这个目标柱做逻辑回归
    lr = lm.LogisticRegression()  # 开始做逻辑回归模型
    lr.fit(X, y)  # 把数据和目标加到模型里
    print(lr.coef_)
    print(lr.intercept_)
    coef_list = lr.coef_[0]  # 系数list
    intercept = lr.intercept_[0]  # 截距

    scorecard_test(coef_list, intercept, feature_list, data_woe_numerical_dict)  # 测试模型


def scorecard_test(coef_list, intercept, feature_list, data_woe_numerical_dict):
    app_train = pd.read_csv('input/application_train.csv')  # 吧testing data变成dataframe
    test_data = app_train[feature_list]  # 要用的特徵

    test_data_woe = replace_with_woe(test_data, data_woe_numerical_dict)  # 把特徵的数字变成woe

    n = len(feature_list)  # 几个特徵
    a = intercept  # 截距
    factor = 28.8539
    offset = 100

    score_list = []  # 分数list

    for index, row in test_data_woe.iterrows():  # 算每一行
        sum_score = 0  # 最后要算出来的分数
        for j in range(len(row)):  # 把每个特徵的分算了再加到一起
            sub_score = -((row[j] * coef_list[j]) + (a/n)) * factor + (offset/n)  # 分数方程
            sum_score += sub_score
        score_list.append(sum_score)  # 把每行算出来的分数加到分数list里

    # 把分数都写到一个文件里
    score_file = open('score_output.txt', 'w')
    for item in score_list:
        score_file.write("%s\n" % item)


def replace_with_woe(og_data, data_woe_numerical_dict):
    # 把数字都cast成float要不然一会儿的woe会被cast成int
    data_replaced_with_woe = og_data.astype(np.float64)

    for feature in data_replaced_with_woe:
        woe_dict = data_woe_numerical_dict[feature]  # 找dictionary里对的特徵
        for i in data_replaced_with_woe.index:
            x = data_replaced_with_woe.at[i, feature]
            for key in woe_dict:
                if key[0] <= x < key[1]:  # 找dictionary里对的bin
                    data_replaced_with_woe.at[i, feature] = woe_dict[key][0]  # 变成woe
                    break
    return data_replaced_with_woe


def get_woe_numerical_dict(data, target):
    #  instantiate WOE dictionary, with type hints
    woe_dict: Dict[Any, Dict[Tuple[Union[float, Any], Union[float, Any]], Tuple[float, Union[float, Any]]]] = {}
    for feature in data.columns[0:len(data.columns)-1]:  # 给每个特徵算woe
        single_feature_df = data.copy()  # .copy() -> 深拷贝dataframe所以不会不小心把原数据改变了
        single_feature_df = single_feature_df[[feature, target]]  # 只把这个特徵和目标加到这个dataframe
        single_feature_df = single_feature_df.sort_values(by=[feature])  # 按数字顺序排序
        value_counts = single_feature_df['TARGET'].value_counts()
        total_non_events = value_counts[0]  # 几个‘0’
        total_events = value_counts[1]  # 几个‘1’

        # TODO Find proper # of bins. Algorithmically算bin多大是最好的
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html
        bins = np.array_split(single_feature_df, 10)  # 分成10个bin (不一定是最好的)
        null_bin = pd.DataFrame()  # 把null values放在这个bin

        bin_dict = {}  # dictionary of woe bins
        bin_num = 0
        smallest_min = single_feature_df[feature].min()  # 这个特徵最小的value
        largest_max = single_feature_df[feature].max()  # 这个特徵最大的value
        for my_bin in bins:
            bin_num += 1

            # TODO Check if these will actually catch NULL/NaN/None values. Have not tested - 还没测试呢
            null_bin = pd.concat([null_bin, my_bin.loc[my_bin[feature].isnull()]], ignore_index=True)
            my_bin = my_bin.drop(my_bin[my_bin[feature].isnull()].index)  # Removes null values from bin

            # 要让scorecard最小和最大的bin包括所有的outliers所以把range增到无穷
            if my_bin[feature].min() != smallest_min:
                bin_min = my_bin[feature].min()  # bin里最小的value吗？
            else:
                bin_min = float('-inf')
            if my_bin[feature].max() != largest_max:
                bin_max = my_bin[feature].max()  # bin里最大的value吗？
            else:
                bin_max = float('inf')

            bin_value_counts = (my_bin['TARGET']).value_counts()
            non_events = (bin_value_counts[0])  # 这个斌有几个non-events
            events = (bin_value_counts[1])  # 这个斌有几个events
            percent_non_events = non_events / total_non_events  # non-events百分之多少在这个bin
            percent_events = events / total_events  # events百分之多少在这个bin
            woe = math.log(percent_non_events / percent_events)  # 算woe
            IV = (percent_non_events - percent_events) * woe  # 算IV

            bin_dict[(bin_min, bin_max)] = (woe, IV)  # 把这个bin的minimum,maximum,woe,和IV存到dictionary里

        woe_dict[feature] = bin_dict
    return woe_dict


if __name__ == '__main__':
    main()
