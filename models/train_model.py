import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn import metrics, preprocessing

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import random
import multiprocessing as mp
from tqdm import tqdm


df = pd.read_csv("/Supplement Sales Prediction/train_folds.csv")
df_test = pd.read_csv("/Supplement Sales Prediction/test.csv")
sample_submission = pd.read_csv("/Supplement Sales Prediction/sample_submission.csv")

def feature_engg(df, df_test):
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

    return df, df_test


df, df_test = feature_engg(df, df_test)

useful_features = [c for c in df.columns if c not in ('ID', 'Sales', 'kfold', 'Date', "#Order")]
object_cols = [col for col in useful_features if df[col].dtype=='object']


### RandomForest Model
final_predictions_rf = []
for fold in range(5):
    print(f"_______________________FOLD:{fold}_______________")
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    ytrain = xtrain.Sales
    yvalid = xvalid.Sales
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    xtest = xtest[useful_features]
    
    oe = preprocessing.LabelEncoder()
    for col in object_cols:
      xtrain[col] = oe.fit_transform(xtrain[col])
      xvalid[col] = oe.transform(xvalid[col])
      xtest[col] = oe.transform(xtest[col])
    
    model = RandomForestRegressor(
        random_state=random.randint(1000,9999), # random_seed = np.random(fold)
        n_jobs = mp.cpu_count()
    )
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions_rf.append(test_preds)
    msle = metrics.mean_squared_log_error(yvalid, preds_valid)*1000
    print('RMSE for fold {} : {}'.format(fold, msle))

### XGB Model
final_predictions_xgb = []
for fold in range(5):
    print(f"_______________________FOLD:{fold}_______________")
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    ytrain = xtrain.Sales
    yvalid = xvalid.Sales
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    xtest = xtest[useful_features]
    
    oe = preprocessing.LabelEncoder()
    for col in object_cols:
      xtrain[col] = oe.fit_transform(xtrain[col])
      xvalid[col] = oe.transform(xvalid[col])
      xtest[col] = oe.transform(xtest[col])
    
    model = XGBRegressor(
        random_state=random.randint(1000,9999),
        tree_method = 'gpu_hist',
        gpu_id = 0,
        predictor = 'gpu_predictor')
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions_xgb.append(test_preds)
    msle = metrics.mean_squared_log_error(yvalid, preds_valid)*1000
    print('MSLE for fold {} : {}'.format(fold, msle))

### CatBoost Model
final_predictions_catboost = []
for fold in range(5):
    print(f"_______________________FOLD:{fold}_______________")
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    ytrain = xtrain.Sales
    yvalid = xvalid.Sales
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    xtest = xtest[useful_features]

    oe = preprocessing.LabelEncoder()
    for col in object_cols:
      xtrain[col] = oe.fit_transform(xtrain[col])
      xvalid[col] = oe.transform(xvalid[col])
      xtest[col] = oe.transform(xtest[col])
    
    model = CatBoostRegressor(random_state=random.randint(1000,9999), task_type = 'GPU', devices='0:1')
    model.fit(xtrain, ytrain, early_stopping_rounds=300, verbose=100,cat_features=object_cols)
    preds_valid = model.predict(xvalid)
    print(preds_valid)
    test_preds = model.predict(xtest)
    final_predictions_catboost.append(test_preds)
    print("Important_features :",sorted(zip(model.feature_importances_,xtrain.columns),reverse=True)[:10])
    msle = metrics.mean_squared_log_error(yvalid, preds_valid)*1000
    print('MSLE for fold {} : {}'.format(fold, msle))

### Light GBM Model
final_predictions_lgbm = []
scores = []
for fold in range(5):
    print(f"_______________________FOLD:{fold}_______________")
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    ytrain = xtrain.Sales
    yvalid = xvalid.Sales
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    xtest = xtest[useful_features]
    
    oe = preprocessing.LabelEncoder()
    for col in object_cols:
      xtrain[col] = oe.fit_transform(xtrain[col])
      xvalid[col] = oe.transform(xvalid[col])
      xtest[col] = oe.transform(xtest[col])
    
    param = {'colsample_bytree': 0.4,
            'learning_rate': 0.008,
            'max_depth': 10,
            'min_child_samples': 118,
            'min_data_per_groups': 36,
            'num_leaves': 690,
            'reg_alpha': 0.010035222189157882,
            'reg_lambda': 4.248793785862258,
            'subsample': 1.0}

    model = LGBMRegressor(
        random_state=random.randint(1000,9999),
        num_iterations=1000,
        n_estimators=500,
        eval_metric='rmse',
        n_jobs=-1,
        **param)
    
    model.fit(xtrain, ytrain, eval_set=(xvalid, yvalid),early_stopping_rounds=300, verbose=100,categorical_feature =object_cols)
    preds_valid = model.predict(xvalid)
    print("Important_features :",sorted(zip(model.feature_importances_,xtrain.columns),reverse=True)[:10])
    msle = metrics.mean_squared_log_error(yvalid, preds_valid)*1000
    print('MSLE for fold {} : {}'.format(fold, msle))
    scores.append(msle)
    
    test_preds = model.predict(xtest)
    final_predictions_lgbm.append(test_preds)

preds_rf = np.mean(np.column_stack(final_predictions_rf), axis=1)
preds_xgb = np.mean(np.column_stack(final_predictions_xgb), axis=1)
preds_catboost = np.mean(np.column_stack(final_predictions_catboost), axis=1)
preds_lgbm = np.mean(np.column_stack(final_predictions_lgbm), axis=1)

preds = (preds_rf + preds_xgb + preds_catboost + preds_lgbm)/4

sample_submission.Sales = preds

sample_submission.to_csv("/Supplement Sales Prediction/submission_file.csv", index=False)