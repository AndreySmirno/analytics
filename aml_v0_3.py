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
import os
import warnings
import pmdarima.arima as ar


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
                            # params_grid[i] = values of params
                            # print('Params grid ', params_grid[i], '   Best param ', best_params, ' i ', i)
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
                                # print('Params grid ', params_grid[i], '   Best param ', best_params)
                                value = (best_params[i] + smax(params_grid[i])) / 2
                                if value > 0:
                                    if (i == 'colsample_bytree') or (i == 'subsample'):
                                        if (best_params[i] + (best_params[i] - value)) > 1:
                                            para.append([value, best_params[i], 1.])
                                        else:
                                            para.append([value, best_params[i], best_params[i] + (best_params[i] - value)])
                                    else:
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
                            # print('Params grid ', params_grid[i], '   Best param ', best_params)
                            if (i == 'colsample_bytree') or (i == 'subsample'):
                                value_less = (params_grid[i][idx] + params_grid[i][idx - 1]) / 2
                                value_more = (params_grid[i][idx + 1] + params_grid[i][idx]) / 2
                                if value_less < 0.:
                                    value_less = 0.
                                if value_more > 1.:
                                    value_more = 1.
                            else:
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
            # best params  {'eta': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 1000, 'reg_lambda': 20.0, 'subsample': 0.5}  new params  {'eta': [0.1, 0.3], 'learning_rate': [0.1, 0.2], 'max_depth': [2, 5, 8], 'n_estimators': [1000, 3000], 'reg_lambda': [20.0], 'subsample': [0.5]}
            print("best params blender", best_params, " new params blender", new_params)
            return new_params, params_grid, score, best_params

        return wrap

    def iterator_params_blender(func):
        def wraper(self, *args, **kwargs):
            best_param = kwargs['dict_arg']
            print('iterator ', best_param)
            after_search = []
            prev_param = []
            score_l = []
            while True:
                best_parm, prev_parm, scoree, params_after = func(self,
                                                                  dict_arg = best_param,
                                                                 )
                # print('iterator part ', best_param)

                log = {
                    'n_estimators': [0, np.inf],
                    'max_depth': [1, np.inf],
                    'eta': [0.00001, 1.],
                    'subsample': [0, 1],
                    'reg_lambda': [0, np.inf],
                    'learning_rate': [0.0000001, 1]
                      }

                for i in best_parm:
                    for j in log:
                        if i == j:
                            if best_parm[i][0] < log[j][0]:
                                print('now ', best_parm[i][0], ' parameter ', i)
                                best_parm[i][0] = log[j][0]
                                print('become ', best_parm[i][0])
                            if best_parm[i][-1] > log[j][-1]:
                                print('now ', best_parm[i][-1])
                                best_parm[i][-1] = log[j][-1]
                                print('become ', best_parm[i][-1], ' parameter ', i)
                stop_rule = []
                cnt_params = 0
                for param in best_parm:
                    if prev_parm[param].dtype() == float:

                        if round(prev_parm[param], 5) == round(best_parm[param], 5):
                            stop_rule.append(1)
                        cnt_params += 1
                    else:
                        if round(prev_parm[param], 5) == round(best_parm[param], 5):
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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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

def autoregressive_forecasting(data, period: int, feature: str):
    X_grid_hi = data[[feature]].rolling(period).mean()
    X_grid_hi[feature+'_predict'] = data[feature].shift(-period).rolling(period).mean()
    X_grid_hi['day'] = data['day']
    X_grid_hi['day_shift'] = data['day'].shift(-(period))
    # X_grid_hi = X_grid_hi.set_index('day_shift')
    # print(X_grid_hi)
    X_grid_hi = X_grid_hi.dropna()
    # print(data['symbol_name'])
    train = round(len(X_grid_hi) - len(X_grid_hi) * 0.5)
    predictions = []

    for i in range(train, len(X_grid_hi)):
        train_set_x = X_grid_hi[feature][:i]
        test_set_y = X_grid_hi[feature+'_predict'][i]
        day_have_shifted = X_grid_hi['day_shift'][i]
        day_have = X_grid_hi['day'][i]



        adf = ar.ADFTest(alpha=0.05)

        if adf.should_diff(train_set_x)[1]:
            # print('series is stationary')
            k = 0
        else:
            # print('nothing')
            k = 0
            while adf.should_diff(train_set_x.loc[k:])[1] != True:
                # print(adf.should_diff(train_set_x.loc[k:])[1])
                k += 1
        # print(train_set_x[k:])
        arima_model = ar.auto_arima(train_set_x[k:], start_p=0,
                                    d=0, start_q=0, max_p=30,
                                    max_q=30, max_d=30,
                                    stepwise=True,
                                    # trace=True, supress_warning=True,
                                    maxiter=100, method="lbfgs")

        preds, conf_int = arima_model.predict(n_periods=1, return_conf_int=True)
        result = pd.DataFrame(preds, columns=['prediction'])
        result = reset_i(result)
        result['real_value'] = test_set_y
        result['day'] = day_have
        result['day_shift'] = day_have_shifted
        predictions.append(result)

    return reset_i(pd.concat(predictions))