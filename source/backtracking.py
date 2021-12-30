import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
from zipline.api import order, symbol
from zipline.algorithm import TradingAlgorithm

import numpy as np

# data
start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2016, 3, 19)
data = web.DataReader("AAPL", "yahoo", start, end)

data = data[['Adj Close']]
data.columns = ["KOREA"]
data = data.tz_localize("UTC")

print(data.head())

df = web.DataReader("078930.KS", "yahoo", start, end)
df = df[['Adj Close']]
df.columns = ['KOREA']

print(df.head())

data = data[len(data) - len(df):]  #데이터 프레임의 row 수를 맞추는 작업
data['KOREA'] = np.where(1, df['KOREA'], df['KOREA'])

print(data.head())

def initialize(context):
    pass

def handle_data(context, data):
    order(symbol('KOREA'), 1)

algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
result = algo.run(data)

plt.plot(result.index, result.portfolio_value)
plt.show()

import pandas_datareader.data as web
import datetime
import numpy    as np
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2016,7,15)
data  = web.DataReader("AAPL", "yahoo", start, end)
data = data[['Adj Close']]
data.columns = ['KOREA']
data.head()
data = data.tz_localize("UTC")


df = web.DataReader("078930.KS", "yahoo", start, end)
df = df[['Adj Close']]
df.columns = ['KOREA']
print(df.head())

data = data[len(data) - len(df):]  #데이터 프레임의 row 수를 맞추는 작업

data['KOREA'] = np.where(1, df['KOREA'], df['KOREA'])    #DB에서 가져온 df의 CLOSE 컬럼을 AAPL의 종가 컬럼으로 교체
print(df.head())
print('***********')
print(data.head())
