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
