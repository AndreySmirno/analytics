import pandas as pd
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.tsa.stattools as adf
from arch import arch_model
pd.set_option('display.max_columns', 500)

today = dt.date.today() - dt.timedelta(days= 1)
last_n_days = today - dt.timedelta(days= 365)
end_day = today.day
end_month=int(today.month)-1
end_year = today.year
end_date = today.strftime('%d.%m.%Y')

#start_day = last_n_days.day
# start_month = int(last_n_days.month) - 1
# start_year = last_n_days.year
# start_date = last_n_days.strftime('%d.%m.%Y')
start_day = '1'
start_month = '0'
start_year = '2021'
start_date = '01.01.2021'




print(today)
print(last_n_days)
print(start_month)
print(start_year)
print(start_date)

def data_converter(eurusd_init):
    eurusd_init = eurusd_init
    datetime_from_dataset = []
    for i in range(len(eurusd_init['<DATE>'])):
            dataset = str(eurusd_init['<DATE>'][i])+' '+str(eurusd_init['<TIME>'][i])
            datetimes = dt.datetime.strptime(dataset, '%Y%m%d %H:%M:%S')
            datetime_from_dataset.append(datetimes)
    eurusd_init['datetime'] = datetime_from_dataset
    eurusd_quotes = eurusd_init[['datetime', '<CLOSE>', '<OPEN>', '<HIGH>', '<LOW>','<TICKER>']]
    eurusd_quotes.columns = ['datetime', 'close', 'open', 'high', 'low', 'ticker']
    #eurusd_quotes['log_price'] = np.log(eurusd_quotes['close']/eurusd_quotes['close'].shift(1))
    #eurusd_quotes['Volatility'] = eurusd_quotes['log_price'].rolling(30, center=False).std()
    #eurusd_quotes['Volatility'] = (eurusd_quotes['Volatility']*30**(1/2))*100
    #eurusd_quotes['mean'] = eurusd_quotes['log_price'].rolling(30, center=False).mean()
    #eurusd_quotes = eurusd_quotes.dropna()
    #eurusd_quotes = eurusd_quotes.reset_index().drop('index', axis=1)
    return eurusd_quotes

def reset_i(data):
    data = data.reset_index()
    data = data.drop('index', axis=1)
    return data

# "https://export.finam.ru/export9.out?market=1&em=3&token=03AGdBq25IV--S9KJG158ltwZ3oNl8CPl8vH84AQZPtgPnByqZ7_nuoWfoFGSlkeVc5OIYSawzKztPPNbJKctduME3EhSm7JcrLFHMQa0TP3EhdQRCB4IVTY2-rt8tv-HOQlKLdV9fPIct5wg4b5pjzsSk-e1LMab7EWL_3OVKnCR29u9HezkfoiFdXwQ4i9WweLNr6OGtPd1xeoGnf8eZlagu-Qm8wXOlcCb_-62W78vtfFf-4LOlVWxaBJ_6S9IuIzFlo6i6cnpd7gnboYB3k5_G_kOWQF-OYOHlhwzlZgO-9AgIeaqq2pzRRJy_5IEdOfXOiJg8DeupoU5iYkILyvBvB70zbrSRmzNZvydQHMnHIVfHLYScOIizXB5M92wavhHizqL6TV6tWk0FBg7NtBWzTTZmT81xn5FHTmKLfQeirayZRjp1cEScSB6NoBXxqY9tq_9ON4khboUDTNs2UaOa5df5WQoAfQ&code=SBER&apply=0&df=1&mf=3&yf=2020&from=01.04.2020&dt=30&mt=5&yt=2020&to=30.06.2020&p=2&f=SBER_200401_200630&e=.txt&cn=SBER&dtf=4&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1"
sber = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=3&token=03AGdBq26yq3eHxCH8SXsJ8FRVIz0PJpD7mHHo-vumtYqRiPX6lwznp6gnJ_egXuIse-tpKao9df7U2XQQEYDCmtP4OvXWJL9UYPybOXat3J0-TlJJC-JWUFFow7CWce-7v2p99Py5kPxG1vT4QUs-urFtp3mrOLd_XfxFwA9Ca5XXoEwX6NCvPkoPOE-M6Pjx7TWH9_KUoQFKdSdgYUkFiVRX2D22QYRFP_-JzeNbtQoMO0wVRwmutmTFjV08OiOgz0qbFgAJXhiGtrY5NluKJEa_xIGOrhymRG1lmnjf4qTl4LhnDqx71d6PlzCd8UEJCB0DNPOmhsjVGPtWWKI0aRX2vbR3U-vg7uugSFDgGm_Ud7x9OVOazAwIc8s3RNY-9JgTQ6IYHU5xBjuQF4BSJZu_QK2UFJeMiUzdv1Ypm2XNtC1HL43V5y8&code=SBER&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=2&f=SBER_201026_201031&e=.csv&cn=SBER&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
data_converter(sber)

y_data = dt.datetime.strptime(str(sber['<DATE>'][0]), '%Y%m%d').year
sber.to_csv('SBER_'+str(y_data)+'.csv', index=False)
sber_y = data_converter(sber)

years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021']

historical_data = []
for i in range(len(years)):
    data = pd.read_csv('SBER_'+str(years[i])+'.csv', sep=',')
    historical_data.append(data_converter(data))

sber_hist = pd.concat(historical_data)

sber_d = reset_i(sber_hist)

sber_d[['close', 'open', 'high', 'low']].plot()



#cl_op = np.log(sber_d['open']/sber_d['close'].shift(1)).dropna()

def yang_zhang_volatility(data, period=int):
    open_p = data['open']
    close = data['close']
    high_p = data['high']
    low_p = data['low']
    close_open = np.log(open_p/close.shift(1)).dropna()

    cl_op = []
    
    for i in range(period, len(close_open)):
            seq = close_open[i-int(period):i]
            sq_e = np.sum((seq - np.mean(seq))**2) / (len(seq)-1)
            cl_op.append(sq_e)

    #return pd.DataFrame(cl_op)

    open_close = np.log(close/open_p)

    op_cl = []

    for i in range(period+1, len(data)):
        seq = open_close[i - int(period):i]
        sq_e = np.sum((seq - np.mean(seq)) ** 2) / (len(seq) - 1)
        op_cl.append(sq_e)

    #return pd.DataFrame(op_cl)

    rs = ((np.log(high_p/close) * np.log(high_p/open_p) + np.log(low_p/close) * np.log(low_p/open_p)).rolling(period).sum()) / period
    rs = rs.dropna()[2:]

    k = 0.34 / (1.34 + (period+1)/(period-1))

    agg = pd.DataFrame({'open_close': op_cl, 'close_open': cl_op, 'Rodgers-Satchel': rs})

    yz = np.sqrt(agg['close_open'] + k*agg['open_close'] + (1-k)*rs)

    yz = reset_i(yz)
    yz.columns = ['Yang-Zhang Volatility']
    return yz


yz_volatility = yang_zhang_volatility(sber_d, 30)
x = np.log(sber_d['close']/sber_d['close'].shift(1)).dropna()
x.plot()

## Implementation of Shapiro-Wilks test that gives us result of normality testing

W, p = stats.shapiro(x)

print('shapiro-wilk: W={0}, p={1}'.format(W, p))

## Implementation of Kolmogorov-Smirnov test that gives us result of normality testing

D, p = stats.kstest(x, 'norm')

## Implementation of Jarque-Bera test that gives us result of normality testing by assimptote method, best way to test normality of big data. If data assimptotically normal, that means the regression of that data would be assimptotically normal too.
stats.jarque_bera(x)



## ADF test results explain wheather timeseries is stationary (if = 1) or not (if = 0), that result is significant to further implementation of ARMA class models
def ADF_test(data):
    res = adf.adfuller(data, autolag='AIC')

    if res[0] < res[4]['5%']:
        return int(1)
    else:
        return int(0)

## count result of adf test of stationarity to in usage of models
ADF_test(x)

#### arch modeling ####
# points of inclusive data to correct modeling:
# 1) Assumptions of normality - i've got test where better one is a Jarque-Berra to comfirm/reject that assumption
# 2) Create tests for other assumptions to choose what is going to be in disrtibution. ()

def stationary_test(data):
    fuller = adf.adfuller(data, autolag='AIC')
    kpss = adf.kpss(data, regression='ct', nlags='auto')

    if (fuller[0] < fuller[4]['5%']) and (kpss[0] < kpss[3]['5%']):
        return int(1)
    else:
        return int(0)


stationarity_res = stationary_test(x)

arch_m = arch_model(x, vol='GARCH', p=1, q=1, dist='Normal')
garch = arch_m.fit(disp='off')
garch_volatility = np.sqrt(garch.params['omega'] + garch.params['alpha[1]'] * garch.resid**2 +
                                             garch.conditional_volatility**2 * garch.params['beta[1]'])

longterm_volty = np.sqrt(garch.params['omega'] / (1-garch.params['alpha[1]']-garch.params['beta[1]']))

volty_df = yz_volatility
volty_df['garch_volty'] = garch_volatility
volty_df['longterm_volty'] = longterm_volty
volty_df.plot()
plt.figure(figsize=(16, 6))
yz_volatility.plot()
garch_volatility.plot()
plt.title('rolling volatility of 30 periods window vs prediction on GARCH(1,1)')
plt.legend()
plt.show()

def rmse_tr(prediction, target):
    return np.sqrt(((prediction - target)**2).mean())





##### GARCH custom estimation #####
# GARCh model is just a way of optimizing raw parameters such as return and volatility with maximizing likelihood
# So optimization of the method shouldn't include model that already built, and parameters could be optimised in another way
# One of the approach is to optimize model with NN with optimization of forecasted parameter of volatility (but it's not meant that parameters will be weights with convergance aruond 1)

## lets try naive approach

import tensorflow.keras as keras

def batches(data, train=float, val=float):
    le = len(data)
    train_le = int(le * train)
    val_le = int(le * val)

    val = train_le+val_le

    train = data[:train_le]

    validate = data[train_le:val]

    test = data[val:]
    print('train :', len(train), 'val :', len(validate), 'test :', len(test))

    return train, validate, test

data_frame = pd.concat([yz_volatility[['Yang-Zhang Volatility']]**2, reset_i(x[30:])**2], axis=1)

data_frame['y'] = data_frame['close'].shift(-1)
data_frame['garch_est'] = reset_i(garch_volatility[30:]**2)

data_frame.plot()
score = rmse_tr(data_frame['garch_est'], data_frame['close'])

naive_train, naive_val, naive_test = batches(data_frame, 0.7, 0.2)

train_set = naive_train[['close', 'garch_est', 'y']].values


def setting(data, window):
    l = len(data)
    w = int(window)
    X_train = []
    y_train = []
    for i in range(w, len(data)):
        X_train.append(data[i-w:i, :])
        y_train.append(data[i, -1])

    return np.array(X_train), np.array(y_train)

X_n_train, y_n_train = setting(train_set, 1000)
len(X_n_train)
len(y_n_train)

# validating_set = np.array(naive_val[['close', 'garch_est', 'y']])
# test_set

# X_n_train = np.array(naive_train[['close', 'garch_est']])
# y_n_train = np.array(naive_train['y'])

# X_n_val = np.array(naive_val[['close', 'garch_est']])
# y_n_val = np.array(naive_val['y'])

#X_nn_train = np.reshape(X_n_train, (X_n_train.shape[0], X_n_train.shape[1], 2))

####### re-modeling GARCH ########

regression = keras.models.Sequential([
    keras.layers.LSTM(units=10, return_sequences=True, input_shape=(X_n_train.shape[1], 3)),
    keras.layers.Dropout(0.1),
    keras.layers.LSTM(units=10, return_sequences=True),
    keras.layers.Dropout(0.1),
    keras.layers.LSTM(units=10, return_sequences=True),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(units=1)
])

regression.compile(optimizer='adam', loss='mean_squared_error')
regression.fit(X_n_train, y_n_train, epochs=100, batch_size=32)

print('RMSE of model to target', score)


## стратегия покупать на 5% CVaR GARCH и продавать на 95% CVaR GARCH. Написать симуляцию процесса на минутных свечах
# внутри дня для нескольких активов и попытаться обучить простую модель, для получения параметров







