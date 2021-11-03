# importi   necessary modules

import pygal
import numpy as np
import pandas as pd
import seaborn as sns

#validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split  #splitting
from sklearn.feature_extraction import DictVectorizer



import gc
import kaleido
import pickle
# import xgboost model
import xgboost 

print("XGBoost version:",xgboost.__version__)  # check XGBoost version

#loading the data
df = pd.read_csv('code/input/hotel_bookings.csv')


# Data Preparation
df['country'] = df['country'].fillna('NA')  #preparing the data
df['children'] = df['children'].fillna(0)  # filling missing values
# filling with 0 and converting to int
df['agent'] = df['agent'].fillna(0).astype(int)
df['company'] = df['company'].fillna(0).astype(int)

# Data Splitting
df_full_train, df_test = train_test_split(df, test_size=0.2,random_state=0)
df_train, df_val = train_test_split(df_full_train, test_size=0.25,random_state=0)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.is_canceled.values
y_val = df_val.is_canceled.values
y_test = df_test.is_canceled.values
y_full_train = df_full_train.is_canceled.values


# final features

categorical = [
    'hotel', 'arrival_date_month', 'meal', 'market_segment',
    'distribution_channel', 'customer_type'
]

numerical = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'agent', 'company',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]






del df_train['is_canceled']
del df_val['is_canceled']
del df_test['is_canceled']
del df_full_train['is_canceled']


dv = DictVectorizer(sparse=False)

train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)



# defining final params for xg_boost
xgb_params = {
        'use_label_encoder':False,
           'learning_rate': 0.25,  # also called eta
             'min_child_weight': 7,
            'gamma': 0.2,
            'max_depth':15,
            'subsample':0.5, #
            'n_estimators':5000, # max_amount of trees
            'colsample_bytree': 0.7,
            'eval_metric':'auc', # area under auc curve as a metric
            'verbosity':2}



print('training the final model...')
# creating a dictionary of features
dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(dicts)
eval_set = [(X_train, y_train),(X_val, y_test)]


model_xgboost = xgboost.XGBClassifier(**xgb_params)

model_xgboost.fit(X_full_train,
                y_full_train,
                early_stopping_rounds=10,
                eval_set=eval_set,
                verbose=False)
    
    

            
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred





y_pred = predict(df_test, dv, model_xgboost)


auc = roc_auc_score(y_test, y_pred)

print(f'Final auc score ={auc}')


#saving the model
output_file = f'code/XGB_model_1.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_xgboost), f_out)

print(f'the model is saved to code/{output_file}')