import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics, preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


df = pd.read_csv("/content/drive/MyDrive/Hackathon/Supplement Sales Prediction/train_folds.csv")
df_test = pd.read_csv("/content/drive/MyDrive/Hackathon/Supplement Sales Prediction/test.csv")
sample_submission = pd.read_csv("/content/drive/MyDrive/Hackathon/Supplement Sales Prediction/sample_submission.csv")

useful_features = [c for c in df.columns if c not in ('ID', 'Sales', 'kfold', '#Order', 'Date')]
object_cols = [col for col in useful_features if df[col].dtype=='object']
df_test = df_test[useful_features]


final_predictions_xgb = []
for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    ytrain = xtrain.Sales
    yvalid = xvalid.Sales
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    le = preprocessing.LabelEncoder()
    xtrain[object_cols] = le.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = le.transform(xvalid[object_cols])
    xtest[object_cols] = le.transform(xtest[object_cols])
    
    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions_xgb.append(test_preds)
    msle = metrics.mean_squared_log_error(yvalid, preds_valid)*1000
    print('RMSE for fold {} : {}'.format(fold, msle))

preds_xgb = np.mean(np.column_stack(final_predictions_xgb), axis=1)

sample_submission.Sales = preds_xgb

sample_submission.to_csv("/Supplement Sales Prediction/sol_baseline_xgb.csv", index=False)