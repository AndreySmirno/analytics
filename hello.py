class range4:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        return RangeIterator(self.start, self.stop)

class RangeIterator:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.stop:
            res = self.start
            self.start += 1
            return res
        raise StopIteration

r = range(0, 100)
assert not hasattr(r, "__contains__")
assert 42 in r

r = range(0, 100)
sum(r)

it = iter(r)
sum(it)

##

def range1(start, stop):
    def step():
        nonlocal start
        res = start
        start += 1
        return res

    return iter(start, stop)


def range2(start, stop):
    while start < stop:
        yield start
        start += 1
        print(start)

print(range2(0, 10))

def g():
    print('started')
    x = 42
    yield x
    print('yield once')
    x += 1
    yield x
    print('yielded twice, done')

it = g()
for x in it:
    print(x)

it = range2(0, 10)

for i in it:
    print(i)

def unique(xs):
    seen = set()
    for item in xs:
        if item in seen:
            continue
        seen.add(item)
        yield item

xs = [1,1,2,3]
assert list(unique(xs)) #== [1,2,3]

# как поместит новое значение в список

def chain(*xss):
    for xs in xss:
        yield xs

xs = [1,2,3]
ys = [92]

assert list(chain(xs, ys))

print(list(chain(xs, ys)))

class BinaryTree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __iter__(self):
        return self.pre_order

    @property
    def pre_order(self):
        yield self
        if self.left:
            yield from self.left
        if self.right:
            yield from self.right

    @property
    def post_order(self):
        return

import pandas as pd
import datetime as dt
import numpy as np

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

def data_converter2(decorate):
    def data_wraper(*args, **kwargs):
        eurusd_init = decorate(*args, **kwargs)
        datetime_from_dataset = []
        for i in range(len(eurusd_init['<DATE>'])):
                dataset = str(eurusd_init['<DATE>'][i])+' '+str(eurusd_init['<TIME>'][i])
                datetimes = dt.datetime.strptime(dataset, '%Y%m%d %H:%M:%S')
                datetime_from_dataset.append(datetimes)
        eurusd_init['datetime'] = datetime_from_dataset
        eurusd_quotes = eurusd_init[['datetime', '<CLOSE>', '<OPEN>', '<HIGH>', '<LOW>','<TICKER>']]
        eurusd_quotes.columns = ['datetime', 'close', 'open', 'high', 'low', 'ticker']
        eurusd_quotes['c_to_c'] = (eurusd_quotes['close']-eurusd_quotes['close'].shift(1)) / eurusd_quotes['close'].shift(1)
        #eurusd_quotes['log_price'] = np.log(eurusd_quotes['close']/eurusd_quotes['close'].shift(1))
        #eurusd_quotes['Volatility'] = eurusd_quotes['log_price'].rolling(30, center=False).std()
        #eurusd_quotes['Volatility'] = (eurusd_quotes['Volatility']*30**(1/2))*100
        #eurusd_quotes['mean'] = eurusd_quotes['log_price'].rolling(30, center=False).mean()
        #eurusd_quotes = eurusd_quotes.dropna()
        #eurusd_quotes = eurusd_quotes.reset_index().drop('index', axis=1)
        return eurusd_quotes
    return data_wraper

def reset_i2(decorate):
    def transform(*args, **kwargs):
        data = decorate(*args, **kwargs)
        data = data.dropna()
        data = data.reset_index()
        data = data.drop('index', axis=1)
        return data
    return transform



# "https://export.finam.ru/export9.out?market=1&em=3&token=03AGdBq25IV--S9KJG158ltwZ3oNl8CPl8vH84AQZPtgPnByqZ7_nuoWfoFGSlkeVc5OIYSawzKztPPNbJKctduME3EhSm7JcrLFHMQa0TP3EhdQRCB4IVTY2-rt8tv-HOQlKLdV9fPIct5wg4b5pjzsSk-e1LMab7EWL_3OVKnCR29u9HezkfoiFdXwQ4i9WweLNr6OGtPd1xeoGnf8eZlagu-Qm8wXOlcCb_-62W78vtfFf-4LOlVWxaBJ_6S9IuIzFlo6i6cnpd7gnboYB3k5_G_kOWQF-OYOHlhwzlZgO-9AgIeaqq2pzRRJy_5IEdOfXOiJg8DeupoU5iYkILyvBvB70zbrSRmzNZvydQHMnHIVfHLYScOIizXB5M92wavhHizqL6TV6tWk0FBg7NtBWzTTZmT81xn5FHTmKLfQeirayZRjp1cEScSB6NoBXxqY9tq_9ON4khboUDTNs2UaOa5df5WQoAfQ&code=SBER&apply=0&df=1&mf=3&yf=2020&from=01.04.2020&dt=30&mt=5&yt=2020&to=30.06.2020&p=2&f=SBER_200401_200630&e=.txt&cn=SBER&dtf=4&tmf=3&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1"

@reset_i2
@data_converter2
def data_reader(path:str, sep:str):
    data = pd.read_csv(path, sep)
    return data


dataset = data_reader("F:/historical_data/daily_for_year/SBER_2020.csv", sep=',')

class data_configurator:
    def __init__(self, path, separator):
        self.path = path
        self.separator = separator

    def data_converter(decorate):
        def data_wraper(*args, **kwargs):
            eurusd_init = decorate(*args, **kwargs)
            datetime_from_dataset = []
            for i in range(len(eurusd_init['<DATE>'])):
                dataset = str(eurusd_init['<DATE>'][i]) + ' ' + str(eurusd_init['<TIME>'][i])
                datetimes = dt.datetime.strptime(dataset, '%Y%m%d %H:%M:%S')
                datetime_from_dataset.append(datetimes)
            eurusd_init['datetime'] = datetime_from_dataset
            eurusd_quotes = eurusd_init[['datetime', '<CLOSE>', '<OPEN>', '<HIGH>', '<LOW>', '<TICKER>']]
            eurusd_quotes.columns = ['datetime', 'close', 'open', 'high', 'low', 'ticker']
            eurusd_quotes['c_to_c'] = (eurusd_quotes['close'] - eurusd_quotes['close'].shift(1)) / eurusd_quotes[
                'close'].shift(1)
            # eurusd_quotes['log_price'] = np.log(eurusd_quotes['close']/eurusd_quotes['close'].shift(1))
            # eurusd_quotes['Volatility'] = eurusd_quotes['log_price'].rolling(30, center=False).std()
            # eurusd_quotes['Volatility'] = (eurusd_quotes['Volatility']*30**(1/2))*100
            # eurusd_quotes['mean'] = eurusd_quotes['log_price'].rolling(30, center=False).mean()
            # eurusd_quotes = eurusd_quotes.dropna()
            # eurusd_quotes = eurusd_quotes.reset_index().drop('index', axis=1)
            return eurusd_quotes

        return data_wraper

    def reset_i2(decorate):
        def transform(*args, **kwargs):
            data = decorate(*args, **kwargs)
            data = data.dropna()
            data = data.reset_index()
            data = data.drop('index', axis=1)
            return data

        return transform

    @reset_i2
    @data_converter
    def data_reader(self: str):
        data = pd.read_csv(self.path, self.separator)
        return data


data_configurator(path="F:/historical_data/daily_for_year/SBER_2020.csv", separator=',').data_reader()