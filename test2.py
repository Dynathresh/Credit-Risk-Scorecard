import pandas as pd

app_train = pd.read_csv('input/application_train.csv')  # Read in training data
feature_list = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']
target = 'TARGET'
train_data = app_train[feature_list]
train_data = train_data.join(app_train[target])

count = 0
for index, row in train_data.iterrows():
    print("index: " + str(index))
    for item in row:
        print(item)
    count += 1
    if count > 5:
        break


for i in range(3):
    print(i)