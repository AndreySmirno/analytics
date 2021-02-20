import yfinance as yf
import datetime as dt
import yaml
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.tsa.stattools as adf
from arch import arch_model

cur_d = dt.date.today()
prev_d = dt.date.today() - dt.timedelta(days=900)

with open('config_currency.yaml') as file:
    symbol_list = yaml.full_load(file)

asset_file = []
for i in range(len(symbol_list['asset'])):
    asset = yf.download(symbol_list['asset'][i], start=str(prev_d), end=str(cur_d), period='1d')
    asset_file.append(asset)

def reset_i(data):
    data = data.reset_index()
    data = data.drop('index', axis=1)
    return data

def yang_zhang_volatility(data, period=int):
    open_p, close, high_p, low_p  = data['open'], data['close'], data['high'], data['low']
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

    rs = ((np.log(high_p / close) * np.log(high_p / open_p) + np.log(low_p / close) * np.log(low_p / open_p)).rolling(
        period).sum()) / period
    rs = rs.dropna()[2:]

    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    agg = pd.DataFrame({'open_close': op_cl, 'close_open': cl_op, 'Rodgers-Satchel': rs})

    yz = np.sqrt(agg['close_open'] + k * agg['open_close'] + (1 - k) * rs)

    yz = reset_i(yz)
    yz.columns = ['Yang-Zhang Volatility']
    return yz


def stationary_test(data):
    fuller = adf.adfuller(data, autolag='AIC')
    kpss = adf.kpss(data, regression='ct', nlags='auto')

    if (fuller[0] < fuller[4]['5%']) and (kpss[0] < kpss[3]['5%']):
        return int(1)
    else:
        return int(0)


## Ljung-Box test is a test for autocorrelation with H0: autocorrelation resids are independent
##                                                   H1: autocorrelation resids are not independent
## In our use-case we assume that residuals are independent and there is no serial autocorrelation in set
## this mean that error have not inherited from previous valuation

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

