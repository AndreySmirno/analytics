
# There is an idea to find out how to implement method guess what kind of volatility is to expect.

#  можно попытаться оценить класс ивента (на уровне ожиданий или нет).
#  + добавить COT методологию в прогноз, как фактор ожидания рынка.
#  тут как раз можно оценить событие по ивенту и набору данных, которые были до него.
#  кандидат LSTM, RNN

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.tsa.stattools as adf
from arch import arch_model
import pmdarima as a
import tensorflow as tf
# для начала хорошо бы продумать как новости реагируют на котировки в день выхода
# algorithm: 1) Eval Volatility by GARCH and forecast by LSTM


class FuncObj:
    def __init__(self, data):
        self.data = data

    def reset_i(self):
        data = self.data.reset_index()
        return data.drop('index', axis=1)

    def data_split(self, train: float, val: float):
        le = len(self.data)
        train_le = int(le * train)
        val_le = int(le * val)
        val = train_le + val_le
        train = self.data[:train_le]
        validate = self.data[train_le:val]
        test = self.data[val:]
        print('train :', len(train), 'val :', len(validate), 'test :', len(test))
        return train, validate, test

    def avg_full_estim_change(self):
        open = self.data['Open']
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        # the way how to convert data into ordinary positive - close to volatility format
        HL = np.log(high / low)
        CO = np.log(close / open)
        CC = np.log(close / close.shift(1))
        prim_data = data.copy()
        prim_data['HL'] = HL
        prim_data['CO'] = CO
        prim_data['CC'] = CC
        ch = (prim_data['HL'] + prim_data['HL'].shift(1) + abs(prim_data['CO']) + abs(prim_data['CO'].shift(1)) + abs(
            prim_data['CC'])) / 5
        data['ch_test'] = ch
        return data.dropna()

    def pct_change(self):
        close = self.data[['Close']]
        return  np.log(close / close.shift(1)).dropna()


class Volatility:
    def __init__(self, data, period: int):
        self.data = data
        self.period = period

    def yang_zhang(self):
        data = self.data
        open_p = data['Open']
        close = data['Close']
        high_p = data['High']
        low_p = data['Low']
        period = self.period
        close_open = np.log(open_p / close.shift(1)).dropna()
        cl_op = []
        for i in range(period, len(close_open)):
            seq = close_open[i - int(period):i]
            sq_e = np.sum((seq - np.mean(seq)) ** 2) / (len(seq) - 1)
            cl_op.append(sq_e)
        open_close = np.log(close / open_p)

        op_cl = []
        for i in range(period + 1, len(data)):
            seq = open_close[i - int(period):i]
            sq_e = np.sum((seq - np.mean(seq)) ** 2) / (len(seq) - 1)
            op_cl.append(sq_e)

        rs = ((np.log(high_p / close) * np.log(high_p / open_p) + np.log(low_p / close) *
               np.log(low_p / open_p)).rolling(period).sum()) / period
        rs = rs.dropna()[2:]
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        agg = pd.DataFrame({'open_close': op_cl, 'close_open': cl_op, 'Rodgers-Satchel': rs})
        yz = np.sqrt(agg['close_open'] + k * agg['open_close'] + (1 - k) * rs)
        yz = yz.reset_index()
        yz.columns = ['Date', 'Yang-Zhang Volatility']
        return yz


class Autoregressive:
    def __init__(self, data):
        self.data = data
    def ADF_test(self):
        res = adf.adfuller(self.data, autolag='AIC')
        if res[0] < res[4]['5%']:
            return int(1)
        else:
            return int(0)

    def stationary_test(self):
        # kpss wanrning is a sign that stationarity confirmed of p-value greater than returned
        i = 0
        result = 0
        while result < 1:
            fuller = adf.adfuller(self.data[i:], autolag='AIC')
            kpss = adf.kpss(self.data[i:], regression='ct', nlags='auto')
            if (fuller[0] < fuller[4]['5%']) and (kpss[0] < kpss[3]['5%']):
                result = int(1)
            else:
                result = int(0)
                i += 1
        return self.data[i:]

    def batching_for_train(self, window: int):
        train_batch = []
        forecast_step_ahead_of_batch = []
        window = window
        for i in range(len(data) - window + 1):
            k = 0
            result = 0
            while result < 1:
                fuller = adf.adfuller(self.data[(i - k):(window + i)], autolag='AIC')
                kpss = adf.kpss(self.data[(i - k):(window + i)], regression='ct', nlags='auto')
                if (fuller[0] < fuller[4]['5%']) and (kpss[0] < kpss[3]['5%']):
                    result = int(1)
                else:
                    result = int(0)
                    k += 1
                    if (i - k) <= 0:
                        window += 1

            model = arch_model(self.data[i:i + window], vol='GARCH', p=1, q=1, dist='Normal')
            garc = model.fit(disp='off')
            garc_volatility = np.sqrt(garc.params['omega'] + garc.params['alpha[1]'] * garc.resid ** 2 +
                                      garc.conditional_volatility ** 2 * garc.params['beta[1]'])
            fore = garc.forecast(horizon=1).variance.iloc[-1:] ** (1 / 2)
            train_batch.append(self.data[i:i + window])
            forecast_step_ahead_of_batch.append(fore)
        forecast_step_ahead_of_batch = pd.concat(forecast_step_ahead_of_batch)
        forecast_step_ahead_of_batch = forecast_step_ahead_of_batch.shift(1)
        forecast_step_ahead_of_batch['X'] = self.data[window:]
        return forecast_step_ahead_of_batch, train_batch


quotes = yf.download('EURUSD=X', start='2015-01-01')

# volty = Volatility(data=quotes, period=24).yang_zhang()


# test_estim_change(quotes)['ch_test'].plot()

ch_norm = FuncObj(quotes).pct_change()
stationary_dt = Autoregressive(ch_norm).stationary_test()
train, val, test = FuncObj(stationary_dt).data_split(train=0.2, val=0.6)
# stationary_dt = \
train_to_garch = Autoregressive(train).stationary_test()

# min batch to estimate
# чего я хочу: чтобы данные за год шли скользящим методом в 1 день и образатывались за 1 тренировочный период назад с прогнозом на 1 день



# next move is to achieve garch estimation as a measure of volatility


# the mechanism of predicting the value over the window is an iterative in function itself realization

# def to_garch_forecast(data, params):
#     data['var'] = data[data.columns[0]]**2
#     data['forecast'] = np.sqrt(params.params['omega'] + params.params['alpha[1]'] * data[data.columns[0]]**2 +
#                                data['var'] * params.params['beta[1]']
#                                )

    # return data, data[['forecast']].shift(1).dropna()




forecasts, init_data_batches = batching_for_train(ch_norm, 600)

forecasts['h.1'].plot()
forecasts.plot()

#
# arch_m = arch_model(train_to_garch, vol='GARCH', p=1, q=1, dist='Normal')
# garch = arch_m.fit(disp='off')
# garch_volatility = np.sqrt(garch.params['omega'] + garch.params['alpha[1]'] * garch.resid ** 2 +
#                            garch.conditional_volatility ** 2 * garch.params['beta[1]'])
# arch_m.fit()
# arch_m.fit()
#
# forecasts = garch.forecast(horizon=5)
# print(forecasts.residual_variance.iloc[-1:]**(1/2))
# print(forecasts.variance.iloc[-1:]**(1/2))
#
#
#
#
#
# garch_volatility.plot()
# to_garch_forecast(train_to_garch, garch)
#
# train_to_garch[['Close', 'forecast', 'forecast_from_box']].plot()
#
# to_garch_tf = train_to_garch
# to_garch_tf['var'] = train_to_garch**2
# to_garch_tf['forecast'] = (np.sqrt(garch.params['omega'] +
#                                   to_garch_tf['Close']**2 * garch.params['alpha[1]'] +
#                                   to_garch_tf['var'] * garch.params['beta[1]'])).shift(1)
#
# to_garch_tf['forecast_from_box'] = garch_volatility
# longterm_volty = np.sqrt(garch.params['omega'] / (1 - garch.params['alpha[1]'] - garch.params['beta[1]']))
#
# garch_volatility.plot()


ff = garch.forecast(horizon=1)
predicted = ff.residual_variance['h.1'].iloc[-1]

ff.mean.iloc[-1].plot()

volty_df = abs(x)
volty_df['garch_volty'] = garch_volatility
volty_df['longterm_volty'] = longterm_volty
volty_df.plot()
plt.figure(figsize=(16, 6))
# tilyz_volaity.plot()
garch_volatility.plot()
plt.title('rolling volatility of 30 periods window vs prediction on GARCH(1,1)')
plt.legend()
plt.show()


def rmse_tr(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())


def garch_validation(data):
    # for SW test p>0.05 = Normality and p<0.05 = NonNormality
    W, p = stats.shapiro(data)

def setting(data, window):
    l = len(data)
    w = int(window)
    X_train = []
    y_train = []
    for i in range(w, len(data)):
        X_train.append(data[i - w:i, :])
        y_train.append(data[i, -1])

    return np.array(X_train), np.array(y_train)

##### GARCH custom estimation #####
# GARCh model is just a way of optimizing raw parameters such as return and volatility with maximizing likelihood
# So optimization of the method shouldn't include model that already built, and parameters could be optimised in another way
# One of the approach is to optimize model with NN with optimization of forecasted parameter of volatility (but it's not meant that parameters will be weights with convergance aruond 1)

## lets try naive approach

import tensorflow.keras as keras


FuncObj(x).data_split(train=0.7, val=0.15)

# idea is to pretrain effective garch(1,1) model that performs well and use that model pattern as general model with linear dependencies


