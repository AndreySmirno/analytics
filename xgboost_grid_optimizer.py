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
        from sklearn.model_selection import GridSearchCV
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

class FuncObj:
    def __init__(self, data):
        self.data = data

    def stand_scale(self):
        import pandas as pd
        df_sc = []
        for i in range(self.data.shape[1]):
            colname = self.data.columns[i]
            res = (self.data[colname] - self.data[colname].mean()) / self.data[colname].std()
            df_sc.append(pd.DataFrame(res, columns=[colname]))
        return pd.concat(df_sc, axis=1)

    def corr_matr(self, data2):
        corr_matrix = []
        for i in range(self.data.shape[1]):
            corrs_i = []
            for j in range(data2.shape[1]):
                corr_i_j = self.data[i].corr(data2[data2.columns[j]])
                corrs_i.append(corr_i_j)
            corr_matrix.append(corrs_i)
        return corr_matrix

    def optimize_n_components(self, factor_threshold: float):
        components_to_fit = 0
        variance_fitted = 0

        while variance_fitted < factor_threshold:
            try:
                pca = PCA(n_components=components_to_fit)
                pca.fit(self.data)
            except:
                print('move next one')
            components_to_fit += 1
            variance_fitted = np.sum(pca.explained_variance_ratio_)
        return components_to_fit, variance_fitted

    def needed_percentile(self):
        import numpy as np
        percent = 0
        cnt_upper = len(data)
        cnt_lower = 0
        while (cnt_upper > cnt_lower):
            cnt_upper = self.data.where(self.data >= np.percentile(self.data, 50+percent))
            cnt_lower = self.data.where(self.data <= np.percentile(self.data, percent))
            perce += 1

        return percent



class QuickSegment:
    def __init__(self, data, factor_threshold: float):
        self.data = data
        self.factor_threshold = factor_threshold

    def auto_rank(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import pandas as pd
        import numpy as np

        scc = StandardScaler()
        scc_d = scc.fit_transform(self.data)
        components_to_fit, \
        variance_fitted = FuncObj(data=scc_d).optimize_n_components(factor_threshold=self.factor_threshold)

        pca_sc = PCA(n_components=components_to_fit)
        pca_sc.fit(scc_d)
        X_sc = pca.fit_transform(scc_d)

        print('Variance fitted: ', variance_fitted)

        importance_top_sc = pd.DataFrame(abs(pca_sc.components_[0]),
                                         columns=['importance']).sort_values(by='importance',
                                                                            ascending=False).head(components_to_fit)

        importance_feature_sc = []
        for i in importance_top_sc.index.astype(int):
            importance_feature_sc.append(seld.data.columns[i])

        stand_data = self.data[importance_feature_sc]

        components = pd.DataFrame(X_sc.reshape(-1, components_to_fit))
        correlation_matrix = FuncObj(components).corr_matr(stand_data)

        direction_est = []
        for i in range(len(correlation_matrix)):
            direction_est.append(np.array(correlation_matrix[i]).T @ pca_sc.explained_variance_ratio_)

        for i in range(components.shape[1]):
            value_r = []
            print(components[components.columns[i]])
            if direction_est[i] > 0:
                for j in range(len(components[components.columns[i]])):
                    if components[components.columns[i]][j] < components[components.columns[i]].mean():
                        value_r.append(0)
                    else:
                        value_r.append(1)
            else:
                for j in range(len(components[components.columns[i]])):
                    if components[components.columns[i]][j] < components[components.columns[i]].mean():
                        value_r.append(1)
                    else:
                        value_r.append(0)
            stand_data['cat_' + str(i)] = value_r
            stand_data['cat_' + str(i)] = stand_data['cat_' + str(i)] * \
                                          (pca_sc.explained_variance_ratio_[i]/sum(pca_sc.explained_variance_ratio_))

        total_est = stand_data['cat_0']

        for i in range(1, components.shape[1]):
            total_est = total_est + stand_data['cat_' + str(i)]

        stand_data['total_est'] = total_est

        return components, stand_data, pca_sc.explained_variance_ratio_, correlation_matrix

    def auto_category(self):
        from xgboost import XGBClassifier
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split

        # method plan: 1) percentiles, 2) split by class, 3) fit the model, 4) learn and validate, 5) categorize arguable data
        pca_comps, ready_data, expl_var, corr_m = self.auto_rank(self.data, self.factor_threshold)

        upper = ready_data[(ready_data['total_est'] >= np.percentile(ready_data['total_est'],
                                                            50+FuncObj(ready_data['total_est']).needed_percentile()))]

        lower = ready_data[(ready_data['total_est'] >= np.percentile(ready_data['total_est'],
                                                            FuncObj(ready_data['total_est']).needed_percentile()))]

        ones_upper = np.ones(len(upper))
        # upper['calss']=ones_upper
        zeros_lower = np.zeros(len(lower))
        # lower['class'] = zeros_lower

        X = np.array(pd.concat([upper, lower]))

        y = np.array(pd.concat([ones_upper, zeros_lower]))

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

        dict_arg = {
            'objective': ['binary:logistic'],
            'subsample': [0.9, 0.8],
            'colsample_bytree': [0.5, 0.4],
            'max_depth': [4, 7],
            'learning_rate': [0.1, 0.0001],
            'gamma': [0.25],
            'reg_lambda': [10., 20., 100.],
            'scale_pos_weight': [1, 3, 6],
            'seed': [42],
            'missing': [1]
        }

        best_params, \
        all_params_est, \
        prev_params, \
        score_of_estimations = LearningAndValidating(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          estimator=XGBClassifier(), quality_func='roc_auc',
                                                          eval_metric='aucpr', cross_val_folds=3
                                                          ).BruteGridCV(dict_arg=params_to_test)

        model = XGBClassifier()
        model.get_params()
        model.set_params(**best_params)
        model.fit(X)

        # and finish pipe till return desired result

        return

if __name__ == '__main__':
    f1_score_for_xgb()
    LearningAndValidating()
    QuickSegment()
