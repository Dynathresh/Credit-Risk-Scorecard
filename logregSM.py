import os
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import math

# import statsmodels.api as sm
#
# print(os.listdir('input'))
# app_train = pd.read_csv('input/application_train.csv')
#
# X = app_train["DAYS_BIRTH"]
# y = app_train["TARGET"]
#
# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(dir(result))
# # print(result.summary())