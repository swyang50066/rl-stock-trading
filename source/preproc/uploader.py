from    datetime    import  datetime

import  pandas      as  pd
import  yfinance    as  yf


def _to_divided_format(date):
    ''' Convert input date format from '%Y%m%d' to '%Y-%m-%d'
    '''
    return str(datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")) 


def _to_collected_format(date):
    ''' Convert input date format from '%Y%-m-%d' to '%Y%m%d'
    '''
    return str(datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d"))


class Uploader(object):
    ''' Upload stock dataset
    '''
    def __init__(self, tickers, start_date, end_date, proxy=None):
        # Set parameters
        self.start_date = _to_divided_format(start_date)
        self.end_date = _to_divided_format(end_date)
        self.tickers = tickers
        self.proxy = proxy

    def select_equal_element_stock(self, df):
        ''' Filter data frame to be listed with tickers 
            that have identical number of elements 
        '''
        # Count number of tickers in the data frame
        series = df.tic.value_counts()
        series = pd.DataFrame(
            {"tic": series.index, "counts": series.values}
        )
        
        # Filter ticker list has equal number of elements
        equals = list(series.counts >= series.counts.mean())
        
        return df[df.tic.isin(series.tic[equals])]

    def fetch(self):
        ''' Fetch stock data from Yahoo API
            7 columns: A date, open, high, low, close, volume and ticker
        '''
        # Define data frame 
        df = pd.DataFrame()

        # Download and save the data in the data frame
        for ticker in self.tickers:
            # Get stock data
            tmp = yf.download(
                ticker, 
                start=self.start_date, 
                end=self.end_date, 
                proxy=self.proxy
            )
            tmp["tic"] = ticker
     
            # Reset the index to use integers as index instead of dates
            tmp = tmp.reset_index()
      
            # Convert the column names to standardized names
            tmp.columns = [
                "date", 
                "open", "high", "low", "close", "adjcp", 
                "volume", 
                "tic",
            ]
            
            # Create day of the week column (monday = 0)
            tmp["day"] = tmp["date"].dt.dayofweek
        
            # Convert date to standard string format, easy to filter
            tmp["date"] = tmp.date.apply(
                lambda item: _to_collected_format(item.strftime("%Y-%m-%d"))
            )
        
            # Use adjusted close price instead of close price
            # so drop the adjusted close price column
            tmp = tmp.drop(labels="close", axis=1)
            
            # Rearrange columns 
            tmp = tmp[[
                "date", "tic", 
                "open", "high", "low", "adjcp", 
                "volume",
            ]]
        
            # Sort items
            tmp = tmp.sort_values(by=["date", "tic"])
            tmp = tmp.reset_index(drop=True)

            # Append to data frame
            df = df.append(tmp)
        
        # Drop missing data
        df = df.dropna()
        df = df.sort_values(by=["date", "tic"])
        df = df.reset_index(drop=True)

        return df
