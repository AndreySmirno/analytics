import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import re
import string

pd.set_option('display.max_columns', 500)

df = pd.read_csv('F:/dataset_samples/Telco_customer_churn.csv')

df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason', 'Count', 'Country', 'State', 'CustomerID', 'Lat Long'], axis=1, inplace=True)

# df['City'].replace('', '_', regex=True, inplace=True)
df.columns = df.columns.str.replace(' ', '_')

## identify missing data
df.dtypes

data_cleared = []
for i in range(len(df['Total_Charges'])):
    print(df['Total_Charges'][i])
    test = re.sub(r'[^0123456789\.]', '', df['Total_Charges'][i])
    data_cleared.append(test)

df['Total_Charges'] = data_cleared

df.loc[(df['Total_Charges'] == ''), 'Total_Charges'] = 0
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])

df.replace(' ', '_', regex=True, inplace=True)

## create datasets to replace data strings to values apropriate to predictions
class data_transformer:
    def __init__ (self, dataset):
        self.dataset = dataset

    def descriptor(self):
        df = self.dataset
        store_set = []
        for column in df:
           uniq = np.unique(df[column])
           print(uniq)


## test regions to accumulate closest by longs and lats
## business part of it is wheather close regions person has friends or mates to bound or similar reasons of usage

X = df.drop('Churn_Value', axis=1).copy()
y = df['Churn_Value'].copy()


X_encoded = pd.get_dummies(X, columns=['City', 'Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service',
                                       'Multiple_Lines', 'Internet_Service', 'Online_Security', 'Online_Backup',
                                       'Device_Protection', 'Tech_Support', 'Streaming_TV',
                                       'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method'])

sum(y) / len(y)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

sum(y_train) / len(y_train) # stratification is a point of data consistency that every and each or our datasets
                            # has the same imbalance of data and training patterns will be the same
                            # the main question - is it so synthetic to stratify if levels variates accross the time

clf_xgb = XGBClassifier(objective='binary:logistic', missing=1, seed=42)
clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])

preds = clf_xgb.predict(X_test)


params_grid = {'max_depth': [3,4,5],
               'learning_rate': [0.1, 0.5, 1.0],
               'gamma' : [0.25],
               'reg_lambda': [10.0, 20, 100],
               'scale_pos_weight': [1,3,5]
               }

opt_params = GridSearchCV(estimator = XGBClassifier(objective='binary:logistic', missing=1, seed=42, subsample=0.9,
                                                    colsample_bytree=0.5),
                          param_grid=params_grid, scoring='roc_auc', verbose=0, n_jobs=-1, cv=3)


opt_params.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_test, y_test)],
               verbose=False)

print(opt_params.best_params_)


clf_xgb = XGBClassifier(objective='binary:logistic', missing=1, seed=42, gamma=0.25, learning_rate=0.1, max_depth=4,
                        reg_lambda=10.0, scale_pos_weight=3, subsample=0.9, colsample_bytree=0.5)

clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])


plot_confusion_matrix(clf_xgb, X_test, y_test, values_format='d', display_labels=['Dod not leave', 'left'])

data_transformer(df).descriptor()
