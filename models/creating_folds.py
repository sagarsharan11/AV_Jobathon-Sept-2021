import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics, preprocessing

df = pd.read_csv("/Supplement Sales Prediction/train.csv")
df_test = pd.read_csv("/Supplement Sales Prediction/test.csv")
sample_submission = pd.read_csv("/Supplement Sales Prediction/sample_submission.csv")

df['kfold'] = -1
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_indices, valid_indices) in enumerate(kf.split(X=df)):
    df.loc[valid_indices, 'kfold'] = fold


df.to_csv("/content/drive/MyDrive/Hackathon/Supplement Sales Prediction/train_folds.csv", index=False)

