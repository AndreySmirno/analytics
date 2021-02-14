import pandas as pd
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
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
start_year = '2020'
start_date = '01.01.2020'




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
    eurusd_quotes = eurusd_init[['datetime', '<CLOSE>', '<TICKER>']]
    eurusd_quotes.columns = ['datetime', 'close', 'ticker']
    eurusd_quotes['log_price'] = np.log(eurusd_quotes['close']/eurusd_quotes['close'].shift(1))
    #eurusd_quotes['Volatility'] = eurusd_quotes['log_price'].rolling(30, center=False).std()
    #eurusd_quotes['Volatility'] = (eurusd_quotes['Volatility']*30**(1/2))*100
    #eurusd_quotes['mean'] = eurusd_quotes['log_price'].rolling(30, center=False).mean()
    eurusd_quotes = eurusd_quotes.dropna()
    eurusd_quotes = eurusd_quotes.reset_index().drop('index', axis=1)
    return eurusd_quotes


sber = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=3&token=03AGdBq26yq3eHxCH8SXsJ8FRVIz0PJpD7mHHo-vumtYqRiPX6lwznp6gnJ_egXuIse-tpKao9df7U2XQQEYDCmtP4OvXWJL9UYPybOXat3J0-TlJJC-JWUFFow7CWce-7v2p99Py5kPxG1vT4QUs-urFtp3mrOLd_XfxFwA9Ca5XXoEwX6NCvPkoPOE-M6Pjx7TWH9_KUoQFKdSdgYUkFiVRX2D22QYRFP_-JzeNbtQoMO0wVRwmutmTFjV08OiOgz0qbFgAJXhiGtrY5NluKJEa_xIGOrhymRG1lmnjf4qTl4LhnDqx71d6PlzCd8UEJCB0DNPOmhsjVGPtWWKI0aRX2vbR3U-vg7uugSFDgGm_Ud7x9OVOazAwIc8s3RNY-9JgTQ6IYHU5xBjuQF4BSJZu_QK2UFJeMiUzdv1Ypm2XNtC1HL43V5y8&code=SBER&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=SBER_201026_201031&e=.csv&cn=SBER&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
moex = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=152798&token=03AGdBq26F0cZDbWT01IBFXwB5VAKnnZx2EEwv52XhdDxXJeozscPeoL0AAGwelBu1QLJmUrEnoC_bAuxHAZKFZt1sA9rW99FNGtHSHJDlu_aLZwqCkRs4a2U81lXNF0Xj2rMEHwv0YkMMlopBR_xKqhDhi_hG9xLln2nDUVVqE7cu5Cs1xKhVqb9HgyQhsLBVWpdyeLpA5P3MeMNgGjfENvS09HQ3ijmCWMaeKDRRi8zM4CTthk19q05Tat18YW9BVu4pHyi74E8JzGhCIeVmnJXBw-dtK2tYx2FbtDgmKW-luLUQNVgoT8NgYS41SJTDx7JsqP-otnNTVuf93BYZv1bLJiaEjjRDj2vYfkTveci-tK7BafpMIG-0QSUdbJcVDx2iGZhORSxnl6FNu30vDJoLSQV-Lmoq6yJa10PJlD4Mc_O5RNozNVE&code=MOEX&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=MOEX_201026_201031&e=.csv&cn=MOEX&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
lkoh = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=8&token=03AGdBq25GF-QsIrPCVkw1PcouErFCCqAXo-Hq-4pmIiqlCxjradc79DP-y-hbMOZpzmltFcOYo9qtPKavQueseH0E4RrJ5W_Cr_ogVn6lEMV_pk6QEldL8W7mZsnvYFVybXjYopXaBWWOm1enjsNqMyeOEnUOiyg0qNAdwTfS95Wnlcp5VFgZkOa154FqdBXBN4nOJE5vKs_zH-m4dniPLIuaA9CCTEUk8n51mIHCrpVsuDgOGshm4DxwWLcrz6L_lhG8-3LSiYkOCjKjho7jhfZIsmhBaV-0Ijje2ZQYzYfFXRZHZjOcBq1Euu4kO26UmLQAS7UsXgJZ-P5vaqw4nMtJlzQUc_kZSt-8fNdzBakbSrWxGZpB8bK_8wmAdPRpmUIBPV0gYKEo9ETW08BwMUcr2Lzro2UD75mmBCEOBsHyrehTSKKVkHw&code=LKOH&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=LKOH_201026_201031&e=.csv&cn=LKOH&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
time.sleep(3)

gazp = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=16842&token=03AGdBq25bPffwlKWBjuKbkxvKSg5WD-CSPyH6mEh4ePvrSrqWY_d6Z-AIy8BWx5YM6wJTbbJDYCT-WewotY1DM4BQdOEqeA-EF1UpwjUrtJzKiiQbGs_hMfhC51AD-bTyoR9a7WDJb135gS0DwT7aUTUlQVddTsxjIfZmU1aUr89_GoX3c2UI8S7updtnNg-4e32h21ETJh5aEKAdkSZckuN-x06l93mpmxWWAG_jS6WtCggdiAYhV31P7GvFxh8M-zmqxRLsIp_wjvSayXHVBUY2ydPhewwex8wCmrOdsBIrJA0XsFBOpwdCMGYFiM303YyKz35bQMvpWFYQJHMlJQsXbYVlboNpodjTxHdh7zFAi8LS2H_yXMS_QL_97EuStAmcIgsGp69XLAl8BxXUSdbTlACrRfenYBLwvGtBPuGb2A3NtzSa9Pg&code=GAZP&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=GAZP_201026_201031&e=.csv&cn=GAZP&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
mvid = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=19737&token=03AGdBq24Bw_5sXssqJtrWZWCQwWNLEn_Tn7jPNBRh66aVtoe7IuRHwiFLfg8WoiFzl_zVlzUcLFAeXWQ-w_uilEJ2r7l8YNSfQ8DkY5k7Vxgoll3OxZfG6rgCCFeuiBBSGUPhFy5HLxmnTdAHX27SJBAWmk5IFXVAlIbnAB1FAUxegQXyCU2CQj_EQ5nUUCisUA-W_suuHQpX-3j42ZtNxskBw6xaDV22cOUaYRBUEuv4u4WCYm0VXsMghFzGDOJnkeJqb-2_GeVBSpI54SeygMKjH550UtA9otACFBOrOUoKerk_DvV-GaJ4ij1aAIyRO17OijGJkIh0cnU1k3Ns62TNDqa2mqWblkHl0clAl9i9Gfp2A8W33FOxipF72P2t8R7QJWsbFV867SKoilbUNY2H4Bp1m4ps1C--bl4APOxMhqSYtisuw6Q&code=MVID&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=MVID_201026_201031&e=.csv&cn=MVID&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
mmk = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=16782&token=03AGdBq25TCQ7mbdS7i02cYeo9z6uffC9XPsQ2x9HEqq-ZDtxWCMRtpxGpV5xqemSBCGiZl0ZcY_GlYQLkAU3H_FgCv767DUKovwSs8Wtr2W6sH45TGMIEc3bvCkSTZxUqiPNGqAb5RBtCrkThNAxSLVJZtLy0WCTJ-1D0kXkroCbcAmuTxLQG_y1bjmrzjAG5C2BBxzKg0JFrrBcIwFBfUHmwJZY_zu8HpTc0B6lPvHig9uCl4N_Ikx_ur61ZzWwsTQXz7nRrrBFYiZvYi1CGsbdqZFcWny4d0EEGtagGPoFENlnaPtf2L6xIsgohr_nKQ_fGPMwdDqsfSFweYzlnuqH9tbK9rZdv5lCeGkuaNdU5cXgyMITAvSCTVeuZhdzPtjU6kHslygPkEpYxQsvfHMCMS7jDIY0IYz3vuMTzZqp3rNzJl9MCblc&code=MAGN&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=MAGN_201026_201031&e=.csv&cn=MAGN&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
time.sleep(3)

magn = pd.read_csv('http://export.finam.ru/export9.out?market=1&em=17086&token=03AGdBq24ZO44MjtKdNwcqlm2RZ34cOJdr9vVvtAKr5Iv9oSEQfI4s17kXMAy1B9zZw3g8-BGsIljHD7bUU_J6t6IfuEJFSatOp9xNPwpl1F1wgvCk5KjxWzs-vYQztAzQWcPBozhoRy56efJGh1jsusDb_oG_FCTiIvheoOGi5uo501clv377sc-ytHyAr9gXDCGk9Px8R0Z3oAekZEnC6WhpwC9Pr-KU5VkV4dvlDPe6zfjYSLDqgFZ9wTdR9czF3tUkdc8n9btC19QZOrXvPUCBDPc5sqOXPTm3LkzXqaRm_SIx0dElXvJj65w2rX9YwlR9ULagI2HvPVYSYJSajmId1oc3bvlkAy059Ks_NiGSyhcRD3sm7Z0aO2omsUTao2KJzRHl3ENw1oD6MtS97n9olkPoOuvq-s0Sk9T0S63dlHKImZJLhqQ&code=MGNT&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=MGNT_201026_201031&e=.csv&cn=MGNT&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
benchmark = pd.read_csv('http://export.finam.ru/export9.out?market=14&em=17455&token=03AGdBq25KBWepANQsMhO0KNMbtKHIv1_S5mFS_La8INNqwgo-DQkOLw4zv65CSyVeKsgbMPw-jCp_kQ795jID1yAvYFAtOJGIRm1lYwdU7nWnCUrJoP80BSgSpxvyf0iiIYci0rwt0zseKrgU6brmDViNrxC5Ol2Alt2cVEhDqvydoXeP0ZQTNg6cMGXg3LfS7oIC1lHzMBQAQAqtBcNQ7IzBH91_xcvfQ8EY8aIgxV9hPdhgFDfo3UdBkXSkACaObdMW7WqDtQWEJn_6rmAcDJnTGKTYmDqjj1i2AXj2mGeos01CsFOjy-XJPqSdub0nBqwIdZR5vfqiNqYcl7v0MLzVkSPFZOf2wWyUe2QJhpM808ioTilnkR0ebUNn0GtWmVoJRQhm-qb2u3-MmY7y9M74-aPRuEbpMha86l0OTKw1p-pewsPGwh4&code=SPFB.RTS&apply=0&df='+str(start_day)+'&mf='+str(start_month)+'&yf='+str(start_year)+'&from='+str(start_date)+'&dt='+str(end_day)+'&mt='+str(end_month)+'&yt='+str(start_year)+'&to='+str(end_date)+'&p=8&f=SPFB.RTS_201026_201031&e=.csv&cn=SPFB.RTS&dtf=1&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1', sep=',')
time.sleep(3)

asset_list = [sber, moex, lkoh, gazp, mvid, mmk, magn] #, benchmark
asset_tickers=['sber', 'moex', 'gazp', 'mvid', 'mmk', 'magn'] #, 'rts_index'
risky_assets = []

for i in range(len(asset_list)):
   change = data_converter(asset_list[i])
   risky_assets.append(change)


#### PORTFOLIO ALLOCATION PART ####

n_portfolios = 10 ** 5
n_days = len(benchmark)

full_asset_df = risky_assets[0]

for i in range(1, len(risky_assets)-1):
        data = risky_assets[i]
        ticker = str(data['ticker'][0])
        data = data[['datetime', 'log_price']]
        data.columns = ['datetime', 'log_price_'+ticker]

        print(data)
        full_asset_df = pd.merge(left=full_asset_df, right=data, how='left', left_on='datetime', right_on='datetime')
        print(full_asset_df)

full_asset_df = full_asset_df.drop(['datetime', 'close', 'ticker'], axis=1)

avg_returns = full_asset_df.mean()
cov_mat = full_asset_df.cov()

np.random.seed(244)

n_assets = len(asset_list) - 1

weights = np.random.random(size=(n_portfolios, n_assets))

weights /= np.sum(weights, axis=1)[:, np.newaxis]

portf_rtns = np.dot(weights, avg_returns)

portf_vol = []

for i in range(0, len(weights)):
    portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))

portf_vol = np.array(portf_vol)
portf_sharpe_ratio = portf_rtns / portf_vol

portf_results_df = pd.DataFrame({'returns': portf_rtns, 'volatility': portf_vol, 'sharpe_ratio': portf_sharpe_ratio})
#portf_res.returns = portf_res.returns

n_points = 100
portf_vol_ef = []
indices_to_skip = []

portf_rtns_eff = np.linspace(portf_results_df.returns.min(), portf_results_df.returns.max(), n_points)

portf_rtns_eff = np.round(portf_rtns_eff, 4)
portf_rtns = np.round(portf_rtns, 4)

for point_index in range(n_points):
    if portf_rtns_eff[point_index] not in portf_rtns:
        indices_to_skip.append(point_index)
        continue
    matched_ind = np.where(portf_rtns == portf_rtns_eff[point_index])
    portf_vol_ef.append(np.min(portf_vol[matched_ind]))

portf_rtns_ef = np.delete(portf_rtns_eff, indices_to_skip)

MARKS = ['$SBER$', '$MOEX$', '$GAZP$', '$MVID$', '$MMK$', '$MGNT$']#,'$RTS$'
fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',
 y='returns', c='sharpe_ratio',
 cmap='RdYlGn', edgecolors='black',
 ax=ax)
ax.set(xlabel='Volatility',
 ylabel='Expected Returns',
 title='Efficient Frontier')
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
for asset_index in range(n_assets):
     ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                 y=avg_returns[asset_index],
                 marker=MARKS[asset_index],
                 s=150,
                 color='black',
                 label=asset_tickers[asset_index])


portf_results_df

max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
min_vol_ind = np.argmin(portf_results_df.volatility)
min_vol_portf = portf_results_df.loc[min_vol_ind]

print('Maximum Sharpe ratio portfolio ----')
print('Performance')

for index, value in max_sharpe_portf.items():
     print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(asset_tickers, weights[np.argmax(portf_results_df.sharpe_ratio)]):
     print(f'{x}: {100*y:.2f}% ', end="", flush=True)


fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',
 y='returns', c='sharpe_ratio',
 cmap='RdYlGn', edgecolors='black',
 ax=ax)
ax.scatter(x=max_sharpe_portf.volatility,
 y=max_sharpe_portf.returns,
 c='black', marker='*',
 s=200, label='Max Sharpe Ratio')
ax.scatter(x=min_vol_portf.volatility,
 y=min_vol_portf.returns,
 c='black', marker='P',
 s=200, label='Minimum Volatility')
ax.set(xlabel='Volatility', ylabel='Expected Returns',
 title='Efficient Frontier')
ax.legend()

import scipy.optimize as sco


def get_portf_rtn(w, avg_rets):
    return np.sum(avg_rets * w)


def get_portf_vol(w, avg_rets, cov_mat):
    return np.sqrt(w.T @ (cov_mat @ w))


def get_efficient_frontier(avg_rets, cov_mat, rets_range):
    eff_portfolio = []
    n_assets = len(avg_returns)
    args = (avg_returns, cov_mat)
    bounds = tuple((0, 1) for assets in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]
    for ret in rets_range:
        constraints = (
            {'type': 'eq', 'fun': lambda x: get_portf_rtn(x, avg_rets) - ret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        eff_portf = sco.minimize(get_portf_vol, initial_guess, args=args, method='SLSQP', constraints=constraints,
                                 bounds=bounds)

        eff_portfolio.append(eff_portf)

    return eff_portfolio


rtns_range = np.linspace(-0.22, 0.32, 200)
efficient_portfolios = get_efficient_frontier(avg_returns, cov_mat, rtns_range)
vols_range = [x['fun'] for x in efficient_portfolios]

fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',
 y='returns', c='sharpe_ratio',
 cmap='RdYlGn', edgecolors='black',
 ax=ax)
ax.plot(vols_range, rtns_range, 'b--', linewidth=3)
ax.set(xlabel='Volatility',
 ylabel='Expected Returns',
 title='Efficient Frontier')

min_vol_ind = np.argmin(vols_range)
min_vol_portf_rtn = rtns_range[min_vol_ind]
min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']
min_vol_portf = {'Return': min_vol_portf_rtn,
 'Volatility': min_vol_portf_vol,
 'Sharpe Ratio': (min_vol_portf_rtn /
 min_vol_portf_vol)}


print('Minimum volatility portfolio ----')
print('Performance')
for index, value in min_vol_portf.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(asset_tickers,
efficient_portfolios[min_vol_ind]['x']):
     print(f'{x}: {100*y:.2f}% ', end="", flush=True)
