import numpy as np
import pandas as pd

import ccxt

# See: 
#   [1] https://wikidocs.net/120392 


class BinanceHistoryAPI(object):
    """To retrieve crypto-currency price information"""
    def __init__(self, symbol="ETH/USDT", timeframe="1d", since=None):
        # Set parameters
        self.api = ccxt.binance()
        self.symbol = symbol
        self.timeframe = timeframe
        self.since = since

    def fetch(self):
        # The since argument is an integer UTC timestamp in milliseconds
        # (everywhere throughout the library with all unified methods).
        eth_ohlcv = self.api.fetch_ohlcv(
            symbol=self.symbol, timeframe=self.timeframe, since=self.since
        )

        # Build data frame
        eth_ohlcv = pd.DataFrame(
            eth_ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )

        # Convert timescale from ms to date
        eth_ohlcv["datetime"] = pd.to_datetime(eth_ohlcv["datetime"], unit="ms")
        eth_ohlcv.set_index("datetime", inplace=True)

        return eth_ohlcv
