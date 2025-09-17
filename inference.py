#!/usr/bin/env python
# coding: utf-8

import numpy as np
import wfdb
import os
import ast
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

# -------------------
# CONFIG
# -------------------
ptbxl_path = "D:\\ECG\\ptb-xl-1.0.3\\"   # Path to PTB-XL dataset
model_path = "D:\\ECG\\models\\First_Paper.h5"

sampling_rate = 100
classification_name = "subclasses"   # {"binary","superclasses","subclasses"}
lead_type = {"lead-I": 1, "bipolar-limb": 3, "unipolar-limb": 3,
             "limb-leads": 6, "precordial-leads": 6, "all-lead": 12}
lead_name = "all-lead"
no_of_leads = lead_type[lead_name]

calssificatin_type = {"binary": 1, "superclasses": 5, "subclasses": 23}
no_of_classes = calssificatin_type[classification_name]

# -------------------
# Load metadata
# -------------------
Y = pd.read_csv(ptbxl_path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

agg_df = pd.read_csv(ptbxl_path + 'scp_statements.csv', index_col=0)
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
    return list(set(tmp))

if classification_name == "superclasses":
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_superclass_diagnostic)
elif classification_name == "subclasses":
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_subclass_diagnostic)
else:
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_superclass_diagnostic)

# -------------------
# Load trained model
# -------------------
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# -------------------
# MultiLabel Binarizer (fit on training labels)
# -------------------
all_labels = Y.diagnostic_superclass.tolist()
mlb = MultiLabelBinarizer()
mlb.fit(all_labels)

# -------------------
# Function to preprocess ECG record
# -------------------
def preprocess_ecg(record_name):
    # Load ECG record
    signal, meta = wfdb.rdsamp(ptbxl_path + record_name)

    # Use all 12 leads (or subset)
    ecg = signal  # shape (1000, 12)

    # Transpose & reshape for CNN
    ecg = ecg.transpose(1, 0)   # (12, 1000)
    ecg = ecg.reshape(1, no_of_leads, 1000, 1)  # (1, 12, 1000, 1)

    return ecg

# -------------------
# Pick a test ECG from dataset
# -------------------
# Example: take the first test sample
test_fold = 10
test_idx = Y[Y.strat_fold == test_fold].index[0]
record_name = Y.loc[test_idx].filename_lr  # filename_hr if sampling_rate=500

print("Testing on record:", record_name)

# -------------------
# Preprocess and Predict
# -------------------
x_sample = preprocess_ecg(record_name)
y_true = Y.loc[test_idx].diagnostic_superclass

y_pred = model.predict(x_sample)
pred_labels = mlb.inverse_transform((y_pred > 0.5).astype(int))

print("\nGround Truth Labels:", y_true)
print("Predicted Labels    :", pred_labels)
