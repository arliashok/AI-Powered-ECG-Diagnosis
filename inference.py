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

classification_name = "subclasses"   # {"binary","superclasses","subclasses"}
lead_type = {"lead-I": 1, "bipolar-limb": 3, "unipolar-limb": 3,
             "limb-leads": 6, "precordial-leads": 6, "all-lead": 12}
lead_name = "all-lead"
no_of_leads = lead_type[lead_name]

calssificatin_type = {"binary": 1, "superclasses": 5, "subclasses": 23}
no_of_classes = calssificatin_type[classification_name]

# -------------------
# Load metadata & labels
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
print("✅ Model loaded successfully!")

# -------------------
# MultiLabel Binarizer (fit on all labels)
# -------------------
all_labels = Y.diagnostic_superclass.tolist()
mlb = MultiLabelBinarizer()
mlb.fit(all_labels)

# -------------------
# Function to preprocess ECG record
# -------------------
def preprocess_ecg(record_name):
    """Load and preprocess ECG record for model input."""
    signal, meta = wfdb.rdsamp(ptbxl_path + record_name)
    ecg = signal  # shape (1000, 12)
    ecg = ecg.transpose(1, 0)   # (12, 1000)
    ecg = ecg.reshape(1, no_of_leads, 1000, 1)
    return ecg

# -------------------
# Prediction Function
# -------------------
def predict_ecg(record_name):
    """Run prediction for a given ECG filename (relative to ptbxl_path)."""
    x_sample = preprocess_ecg(record_name)

    # Find ground truth labels (if available in ptbxl_database.csv)
    row = Y[Y.filename_lr == record_name]
    if row.empty:
        row = Y[Y.filename_hr == record_name]
    if not row.empty:
        y_true = row.iloc[0].diagnostic_superclass
    else:
        y_true = None

    y_pred = model.predict(x_sample)
    pred_labels = mlb.inverse_transform((y_pred > 0.5).astype(int))

    print("\n🔎 Testing record:", record_name)
    if y_true is not None:
        print("Ground Truth Labels:", y_true)
    print("Predicted Labels    :", pred_labels)

# -------------------
# Evaluate Model on Test Set
# -------------------
def evaluate_model():
    print("\n📊 Evaluating model on full test set...")
    # Load saved test set
    x_test = np.load("D:\\ECG\\x_test.npy", allow_pickle=True)
    y_test = np.load("D:\\ECG\\y_test.npy", allow_pickle=True)

    # Reshape
    x_test = x_test.transpose(0, 2, 1)
    x_test = x_test.reshape(x_test.shape[0], no_of_leads, 1000, 1)

    # Binarize labels
    if classification_name != "binary":
        y_test = mlb.transform(y_test)

    # Evaluate
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test Binary Accuracy : {scores[1]:.4f}")
    print(f"✅ Test ROC-AUC Score   : {scores[2]:.4f}")

# -------------------
# Example Usage
# -------------------
if __name__ == "__main__":
    # Predict for a single file
    test_file = "records100/00000/00814_lr"   # << change this to your file name
    predict_ecg(test_file)

    # Evaluate full test set
    evaluate_model()
