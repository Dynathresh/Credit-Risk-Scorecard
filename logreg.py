import pandas as pd
import sklearn.linear_model as lm

app_train = pd.read_csv('input/application_train.csv')
X = app_train[["DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL"]]
# X = X.reshape(-1, 1)
y = app_train["TARGET"]

lr = lm.LogisticRegression()
lr.fit(X, y)
print(lr.coef_)
print(lr.intercept_)
