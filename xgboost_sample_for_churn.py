# disclaimer: This realization is not the most precise one, but automated enought to work for emprovement of precision
#             of searching algorithm, there are several strategies to do so.
#             You can make your own extention of method and report me to see you results.
#             And please give me motivated thing to improve my code, thanks in advise!

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, f1_score, classification_report, make_scorer
import re
import string
import math

pd.set_option('display.max_columns', 500)


def _F1_eval(preds, labels):
    #  Part of code to create each of the threshold availabale for estimation
    t = np.arange(0, 1, 0.005)
    #  Coz of step threshold 0.5%, than we have 200 steps
    f = np.repeat(0, 200)
    results = np.vstack([t, f]).T
    n_pos_examples = sum(labels)
    if n_pos_examples == 0:
        raise ValueError("labels not containing positive examples")

    #  looking for the best one in all range of thresholds
    for i in range(200):
        pred_indexes = (preds >= results[i, 0])
        TP = sum(labels[pred_indexes])
        FP = len(labels[pred_indexes]) - TP
        precision = 0
        recall = TP / n_pos_examples

        if (FP + TP) > 0:
            precision = TP / (FP + TP)

        if (precision + recall > 0):
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0
        results[i, 1] = F1
    # getting max f1 score available
    return (max(results[:, 1]))

# Exact function to implement into XGBoost method as eval_metric
def f1_score_for_xgb(preds, dtrain):
    y_true = dtrain.get_label()
    err = _F1_eval(preds,dtrain.get_label())
    return 'f1', 1-err

class LearningAndValidating:
    ## logically much better to inti core args here and translate then into other methods
    def __init__(self, X_train, y_train, X_test, y_test, estimator, quality_func: str, eval_metric: str,
                        cross_val_folds: int):
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test
        self.estimator = estimator
        self.quality_func = quality_func
        self.eval_metric = eval_metric
        self.cross_val_folds = cross_val_folds


    def params_blender(func):
        def wrap(self, *args, **kwargs):
            params_grid, best_params, score = func(self, *args, **kwargs)

            def sm(num):
                if len(num) == 1:
                    m2 = num[0]
                else:
                    m1 = m2 = float('inf')
                    for x in num:
                        if x <= m1:
                            m1, m2 = x, m1
                        elif x < m2:
                            m2 = x
                return m2

            def smax(num):
                if len(num) == 1:
                    m2 = num[0]
                else:
                    m1 = m2 = float('inf')
                    for x in num:
                        if x >= m1:
                            m1, m2 = x, m1
                        elif x > m2:
                            m2 = x
                return m2

            name = []
            para = []
            for i in best_params:
                if len(params_grid[i]) > 1:
                    if best_params[i] == min(params_grid[i]):
                        name.append(i)
                        if type(best_params[i]) == float:
                            value = (best_params[i] + sm(params_grid[i])) / 2
                            if best_params[i] - (value - best_params[i]) > 0:

                                para.append([best_params[i] - (value - best_params[i]), best_params[i], value])
                            else:
                                para.append([best_params[i], value])
                        else:
                            value = round((best_params[i] + sm(params_grid[i])) / 2)
                            if best_params[i] - (value - best_params[i]) > 0:
                                para.append([best_params[i] - (value - best_params[i]), best_params[i], value])
                            else:
                                para.append([best_params[i], value])
                    elif best_params[i] == max(params_grid[i]):
                        name.append(i)
                        if math.isinf(smax(params_grid[i])):
                            para.append([best_params[i]])
                        else:
                            if type(best_params[i]) == float:
                                value = (best_params[i] + smax(params_grid[i])) / 2
                                if value > 0:
                                    para.append([value, best_params[i], best_params[i] + (best_params[i] - value)])
                                else:
                                    para.append([best_params[i], best_params[i] + (best_params[i] - value)])
                            else:
                                value = round((best_params[i] + smax(params_grid[i])) / 2)
                                if value > 0:
                                    para.append([value, best_params[i], best_params[i] + (best_params[i] - value)])
                                else:
                                    para.append([best_params[i], best_params[i] + (best_params[i] - value)])

                    else:
                        idx = params_grid[i].index(best_params[i])
                        name.append(i)
                        if type(best_params[i]) == float:
                            value_less = (params_grid[i][idx] + params_grid[i][idx - 1]) / 2
                            value_more = (params_grid[i][idx + 1] + params_grid[i][idx]) / 2
                            para.append([value_less, best_params[i], value_more])
                        else:
                            if (params_grid[i][idx + 1] - params_grid[i][idx]) and (
                                    params_grid[i][idx] - params_grid[i][idx - 1]) > 1:
                                value_less = round((params_grid[i][idx] + params_grid[i][idx - 1]) / 2)
                                value_more = round((params_grid[i][idx + 1] + params_grid[i][idx]) / 2)
                                para.append([value_less, best_params[i], value_more])
                            else:
                                para.append([best_params[i]])
                else:
                    name.append(i)
                    para.append([best_params[i]])

            new_params = dict(zip(name, para))
            return new_params, params_grid, score, best_params

        return wrap

    def iterator_params_blender(func):
        def wraper(self, *args, **kwargs):
            best_param = kwargs['dict_arg']
            after_search = []
            prev_param = []
            score_l = []
            while True:
                best_parm, prev_parm, scoree, params_after = func(self,
                                                                  dict_arg = best_param,
                                                                 )
                stop_rule = []
                cnt_params = 0
                for param in best_parm:
                    if prev_parm[param] == best_parm[param]:
                        stop_rule.append(1)
                    cnt_params += 1

                if sum(stop_rule) == cnt_params:
                    break
                else:
                    best_param = best_parm
                    after_search.append(params_after)
                    prev_param.append(prev_parm)
                    score_l.append(scoree)

            return after_search[-1], after_search, prev_param, score_l

        return wraper

    @iterator_params_blender
    @params_blender
    def BruteGridCV(self, dict_arg):
        dict_arg = dict_arg
        opt_params = GridSearchCV(
            estimator=self.estimator,
            param_grid=dict_arg, scoring=self.quality_func, verbose=0, n_jobs=-1, cv=self.cross_val_folds)
        opt_params.fit(self.X_train, self.y_train, early_stopping_rounds=10, eval_metric=self.eval_metric, eval_set=[(self.X_test, self.y_test)],
                       verbose=False)

        best_params = opt_params.best_params_
        best_score = opt_params.best_score_
        return dict_arg, best_params, best_score

    def learning_process(self, model_param_init, eval_metric: str):
        print(model_param_init)
        k = []
        v = []
        for i in model_param_init:
            k.append(i), v.append(model_param_init[i])

        model_init = dict(zip(k, v))
        print(model_init)
        model = self.estimator
        model.get_params()
        model.set_params(**model_init)
        model.fit(self.X_train, self.y_train, verbose=True, early_stopping_rounds=10, eval_metric=eval_metric,
                  eval_set=[(self.X_test, self.y_test)])
        preds = model.predict(self.X_test)
        conf_matr = confusion_matrix(preds, self.y_test)

        result_cf = []
        for i in range(len(conf_matr)):
            metric = conf_matr[i][0] / np.sum(conf_matr[i])
            result_cf.append(metric)

        return preds, self.y_train, conf_matr, result_cf



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


X = df.drop('Churn_Value', axis=1).copy()
y = df['Churn_Value'].copy()



X_encoded = pd.get_dummies(X, columns=['City', 'Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service',
                                       'Multiple_Lines', 'Internet_Service', 'Online_Security', 'Online_Backup',
                                       'Device_Protection', 'Tech_Support', 'Streaming_TV',
                                       'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method'])


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

params_to_test = {
    'objective' : ['binary:logistic'],
    'subsample': [0.9, 0.8],
    'colsample_bytree': [0.5, 0.4],
    'max_depth': [4, 7],
   'learning_rate': [0.1, 0.0001],
   'gamma' : [0.25],
   'reg_lambda': [10., 20., 100.],
   'scale_pos_weight': [1, 3, 6],
    'seed': [42],
    'missing': [1]

}

f1 = make_scorer(f1_score, average='binary')

best_params,\
all_params_est,\
prev_params,\
score_of_estimations = LearningAndValidating(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                estimator=XGBClassifier(), quality_func=f1,
                                eval_metric=f1_score_for_xgb, cross_val_folds=3
                                ).BruteGridCV(dict_arg=params_to_test)

predictions,\
test_y_data,\
confusion_matrix,\
conf_matrix_ratios = LearningAndValidating(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                      estimator=XGBClassifier(), quality_func=f1,
                      eval_metric=f1_score_for_xgb, cross_val_folds=3
                      ).learning_process(model_param_init=best_params, eval_metric=f1_score_for_xgb)


# The result is that tuning of model that way creates a bit better precision over the fine tuning model


# The comon flow with a lot of rounds of tuning parameters to find the best set

test_model = XGBClassifier(gamma=0.25, colsample_bytree=0.5, max_depth=4, seed=42, subsample=0.9,
                           missing=1, objective='binary:logistic',reg_lambda=10, scale_pos_weight=3,
                           learn_rate=0.1)

test_model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='aucpr',
                  eval_set=[(X_test, y_test)])


test_predictions = test_model.predict(X_test)
test_predictions = pd.DataFrame(test_predictions, columns=['predictions'])

plot_confusion_matrix(test_model, X_test, y_test, values_format='d', display_labels=['Dod not leave', 'left'])

print(classification_report(y_test, test_predictions))

#  result of test other person is: 75.12% or less if initially used common loss function (were 74.73%)
#  this automated method gives: 76.43%
#  automated method is works not that good, but faster and still a bit more precise.
#  the main advantage that it works automative and still good for common and custom eval metrics and at test works
#  approximately same. 
