#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import wfdb
import ast
import os


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = 'D:\\ECG\\ptb-xl-1.0.3\\'
sampling_rate = 100
calssificatin_type = "superclasses"  # {"binary","superclasses","subclasses"}

lead_types = {"lead-I": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "bipolar-limb": [3, 4, 5, 6, 7, 8, 9, 10, 11],
              "unipolar-limb": [0, 1, 2, 6, 7, 8, 9, 10, 11], "limb-leads": [6, 7, 8, 9, 10, 11],
              "precordial-leads": [0, 1, 2, 3, 4, 5], "all-lead": []}
lead_name = "all-lead"

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_superclass_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def aggregate_subclass_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    ret = list(set(tmp))
    return ret


if calssificatin_type == "superclasses":
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_subclass_diagnostic)
else:
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_superclass_diagnostic)

# Split data into train and test
test_fold = 10
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

y_train = y_train.tolist()
y_test = y_test.tolist()

if calssificatin_type == "binary":
    count = 0
    for i in y_train:
        if 'MI' in i or 'HYP' in i or 'CD' in i or 'STTC' in i:
            y_train[count] = 0
        elif 'NORM' in i:
            y_train[count] = 1
        else:
            y_train[count] = 0
        count += 1

    count = 0
    for i in y_test:
        if 'MI' in i or 'HYP' in i or 'CD' in i or 'STTC' in i:
            y_test[count] = 0
        elif 'NORM' in i:
            y_test[count] = 1
        else:
            y_test[count] = 0
        count += 1

print("done")


def mkdir_ifnotexists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


# leads I
list_train = []
list_test = []
for i in X_train:
    t = np.delete(i, lead_types[lead_name], 1)
    list_train.append(t)

for i in X_test:
    t = np.delete(i, lead_types[lead_name], 1)
    list_test.append(t)

# Save datasets
np.save('x_train.npy', np.array(list_train))
np.save('x_test.npy', np.array(list_test))
np.save('y_train.npy', np.array(y_train, dtype=object))
np.save('y_test.npy', np.array(y_test, dtype=object))

print("Training and testing data saved successfully!")
