import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt # import datetime, date, timedelta
import yfinance as yf
pd.set_option('display.max_columns', 500)

## insert quotes data for analysis

quotes = pd.read_csv('C:/py_projects_and_stuff/cot/EURUSD_210701_210801.txt', sep=',')

quotes = yf.download('USDJPY=X', start='2021-07-01', end='2021-08-13')


## CFTC COT analysis part ##
y = ['2020', '2021']

# curr_year_data = 'https://www.cftc.gov/sites/default/files/files/dea/history/com_fin_txt_2021.zip'


cot = []
for i in range(len(y)):
    data = pd.read_csv('C:/py_projects_and_stuff/cot/'+y[i]+'_cot.txt', sep=',')

    data = data[['Market_and_Exchange_Names', 'Report_Date_as_YYYY-MM-DD',
                 'Open_Interest_All', 'Asset_Mgr_Positions_Long_All', 'Asset_Mgr_Positions_Short_All']]

    cot.append(data)


cot_ch = pd.concat(cot)


rates = ['EURO FX - CHICAGO MERCANTILE EXCHANGE', 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE',
         'U.S. DOLLAR INDEX - ICE FUTURES U.S.', 'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
         'BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE', 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
         'NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE', 'BITCOIN - CHICAGO MERCANTILE EXCHANGE',
         'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE', 'SOUTH AFRICAN RAND - CHICAGO MERCANTILE EXCHANGE',
         'VIX FUTURES - CBOE FUTURES EXCHANGE']


to_print = []
for i in range(len(rates)):
    part = cot_ch[cot_ch['Market_and_Exchange_Names'] == rates[i]]
    part['net_position'] = part['Asset_Mgr_Positions_Long_All'] - part['Asset_Mgr_Positions_Short_All']
    part = part.sort_values('Report_Date_as_YYYY-MM-DD', ascending=True)


    # part['posit_ch'] = (part['Open_Interest_All'] - part['Open_Interest_All'].shift(1))/part['Open_Interest_All'].shift(1)
    # part['net_ch'] = (part['net_position'] - part['net_position'].shift(1)) / part['net_position'].shift(1)
    part['Q'] = part['net_position'] / part['Open_Interest_All']
    part['cot_indicator'] = (part['Q'] - part['Q'].rolling(20).min()) / (part['Q'].rolling(20).max() - part['Q'].rolling(20).min())
    part = part.dropna()
    part = part.reset_index().drop('index', axis=1)
    to_print.append(part)
    print(part)


ready_cot_analysis = []
for i in range(len(to_print)):
    start = dt.datetime.strptime(to_print[i]['Report_Date_as_YYYY-MM-DD'][-4:-3].values[0], '%Y-%m-%d').date()
    delta = dt.date.today() - start
    days = [start + dt.timedelta(days=i) for i in range(delta.days + 1)]
    day_len = pd.DataFrame(np.array(days), columns=['days'])
    conv_date = []
    for k in range(len(to_print[i])):
        get_val = dt.datetime.strptime(to_print[i]['Report_Date_as_YYYY-MM-DD'][k], '%Y-%m-%d').date()
        conv_date.append(get_val)
    to_print[i]['days'] = conv_date
    ready_cot = pd.merge(left=day_len, right=to_print[i][['days', 'Market_and_Exchange_Names', 'cot_indicator']],
                         how='left', left_on='days', right_on='days')
    ready_cot = ready_cot.fillna(method='ffill')
    ready_cot_analysis.append(ready_cot)


# Currency to pick

cot_to_pick = pd.concat(ready_cot_analysis) #['Market_and_Exchange_Names'] == ['JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE']
pick_report = cot_to_pick[cot_to_pick['Market_and_Exchange_Names'] == 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE']
#######################################################################################################################
# quotes preparation for merge to analysis


quotes = quotes.reset_index()
quotes['Date']

datetime_from_dataset = []
for i in range(len(quotes['Date'])):
    dataset = str(quotes['Date'][i]) # + ' ' + str(eurusd_init['<TIME>'][i])
    datetimes = dt.datetime.strptime(dataset, '%Y-%m-%d %H:%M:%S').date()
    datetime_from_dataset.append(datetimes)

quotes['Date'] = datetime_from_dataset

ready_dataframe_for_plot = pd.merge(left=ready_cot_analysis[0], right=quotes[['Date', 'Close']], how='left',
                                    left_on='days', right_on='Date')
ready_dataframe_for_plot = ready_dataframe_for_plot.dropna()
ready_dataframe_for_plot = ready_dataframe_for_plot.set_index('days')
to_print[0]
days_len[0].columns

plt.plot(ready_dataframe_for_plot[['cot_indicator', 'Close']])
plt.ylabel('index')
plt.xlabel('dates')
plt.show()

ready_dataframe_for_plot[['cot_indicator', 'Close']].to_csv('C:/py_projects_and_stuff/cot/cot_ready/USDJPY_16_08_2021.csv')

# 'EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE',
#          'EURO FX/JAPANESE YEN XRATE - CHICAGO MERCANTILE EXCHANGE',

np.unique(cot_ch['Market_and_Exchange_Names'])