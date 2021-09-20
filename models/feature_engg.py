import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold


df = pd.read_csv("/Supplement Sales Prediction/train_folds.csv")
df_test = pd.read_csv("/Supplement Sales Prediction/test.csv")
sample_submission = pd.read_csv("/Supplement Sales Prediction/sample_submission.csv")

## Changing Date to TimeStamp
df['Date'] = pd.to_datetime(df['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

## Changing Store_id to object
df['Store_id'] = df['Store_id'].astype(str)
df_test['Store_id'] = df_test['Store_id'].astype(str)

useful_features = [c for c in df.columns if c not in ('ID', 'Sales', 'kfold', "#Order")]
object_cols = [col for col in useful_features if df[col].dtype=='object']
numerical_cols = ['Holiday']

## Target Encoding
for col in object_cols:
    temp_df = []
    temp_test_feat = None
    temp_test_order_feat = None
    for fold in range(5):
        xtrain = df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        feat = xtrain.groupby(col)['Sales'].agg('mean')  ### always target encode on the training set and not on the evaluation, valid set
        feat1 = xtrain.groupby(col)['#Order'].agg("mean")
        feat = feat.to_dict()
        feat1 = feat1.to_dict()
        xvalid.loc[:, f"target_enc_{col}"] = xvalid[col].map(feat)
        xvalid.loc[:, f"#order_enc_{col}"] = xvalid[col].map(feat1)
        temp_df.append(xvalid)  ### for every fold a new xvalid data is added to the temp_df
        if (temp_test_feat is None) and (temp_test_order_feat is None):
            temp_test_feat = df_test[col].map(feat)
            temp_test_order_feat = df_test[col].map(feat1)
        else:
            temp_test_feat += df_test[col].map(feat)
            temp_test_order_feat += df_test[col].map(feat1)
    temp_test_feat /= 5
    temp_test_order_feat /= 5
    df_test.loc[:,f"target_enc_{col}"] = temp_test_feat
    df_test.loc[:,f"#order_enc_{col}"] = temp_test_order_feat
    df = pd.concat(temp_df)


## Feature engineering groupby features | combination of categorical and numerical variables
for frame in tqdm([df, df_test]):
  for col in object_cols:
    for cols in numerical_cols:
        frame[cols+"_"+col+"_mean"] = frame.groupby(col)[cols].transform('mean')
        frame[cols+"_"+col+"_median"] = frame.groupby(col)[cols].transform('median')
        frame[cols+"_"+col+"_std"] = frame.groupby(col)[cols].transform('std')
        frame[cols+"_"+col+"_count"] = frame.groupby(col)[cols].transform('count')


## Creating new features
for frame in [df, df_test]:
  # frame.loc[:,'year'] = frame.Date.dt.year
  # frame.loc[:,'month'] = frame.Date.dt.month
  # frame.loc[:,'day'] = frame.Date.dt.day
  frame.loc[:,'weekend'] = (frame.Date.dt.weekday >= 5).astype(int)
  # frame.loc[:,'weekofyear'] = frame.Date.dt.weekofyear
  frame.loc[:,'dayofweek'] = frame.Date.dt.dayofweek

  frame['Holiday'+'_'+'dayofweek'] = frame['Holiday'].map(str) + '_' + frame['dayofweek'].map(str)
  frame['Holiday'+'_'+'weekend'] = frame['Holiday'].map(str) + '_' + frame['weekend'].map(str)
  frame['Holiday'+'_'+'Discount'] = frame['Holiday'].map(str) + '_' + frame['Discount'].map(str)
  frame['Store_Type'+'_'+'Location_Type'] = frame['Store_Type'].map(str) + '_' + frame['Location_Type'].map(str)
  frame['Store_Type'+'_'+'Discount'] = frame['Store_Type'].map(str) + '_' + frame['Discount'].map(str)
  frame['Discount'+'_'+'Location_Type'] = frame['Discount'].map(str) + '_' + frame['Location_Type'].map(str)
  frame['Store_Type'+'_'+'Region_Code'] = frame['Store_Type'].map(str) + '_' + frame['Region_Code'].map(str)
  frame['Discount'+'_'+'Region_Code'] = frame['Discount'].map(str) + '_' + frame['Region_Code'].map(str)
  frame['Location_Type'+'_'+'Region_Code'] = frame['Location_Type'].map(str) + '_' + frame['Region_Code'].map(str)
  frame['Store_Type'+'_'+'Location_Type'+'_'+'Discount'] = frame['Store_Type'].map(str) + '_' + frame['Location_Type'].map(str) + '_' + frame['Discount'].map(str)
  frame['Store_Type'+'_'+'Location_Type'+'_'+'Region_Code'] = frame['Store_Type'].map(str) + '_' + frame['Location_Type'].map(str) + '_' + frame['Region_Code'].map(str)


