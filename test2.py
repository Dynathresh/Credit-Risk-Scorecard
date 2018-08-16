import pandas as pd


app_train = pd.read_csv('input/application_train.csv')  # 把training data变成dataframe

# Find correlations with the target and sort
correlations = app_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))