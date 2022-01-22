### Meaning of Dataset columns

Data frame of the dataset is listed in the order of 
```
dataframe_columns = [
    'datadate', 'tic', 
    'adjcp', 'open', 'high', 'low', 'volume', 
    'macd', 'rsi', 'cci', 'adx', 'turbulence'
]
```

* Raw columns 
  * datadate: Date
  * tic: Ticker
  * prccd: Close Price (Daily)
  * prcod: Open Price (Daily)
  * prchd: High Price (Daily)
  * prcld: Low Price (Daily)
  * ajexdi: Daily Adjustment Factor
  * cshtrd: Common Shares Traded

* To OHLCV (Open-High-Low-Close-Volume) columns
  * adjcp: adjusted close price = prccd / ajexdi
  * open = prcod / ajexdi
  * high = prchd / ajexdi
  * low = prcld / ajexdi 
  * volume = cshtrd

* Technical indicators
  * macd: moving average convergence divergence
  * rsi: relative strength index
  * cci: commodity channel index
  * adx: average directional index
  * turbulence: financial turbulence index

* A 
 
