import  pandas      as  pd
import  yfinance    as  yf


class Uploader(object):
    ''' Upload stock dataset
    '''
    def __init__(self, tickers, start, end, proxy=None):
        # Set parameters
        self.start = start
        self.end = end
        self.tickers = tickers
        self.proxy = proxy

    def selectEqualElementStock(self, df):
        ''' Filter data frame to be listed with tickers 
            that have identical number of elements 
        '''
        # Count number of tickers in the data frame
        countSeries = df.tic.value_counts()
        countSeries = pd.DataFrame(
            {"tic": countSeries.index, "counts": countSeries.values}
        )
        
        # Filter ticker list has equal number of elements
        equals = list(countSeries.counts >= countSeries.counts.mean())
        
        return df[df.tic.isin(countSeries.tic[equals])]

    def fetch(self):
        ''' Fetch stock data from Yahoo API
            7 columns: A date, open, high, low, close, volume and ticker
        '''
        # Define data frame 
        df = pd.DataFrame()

        # Download and save the data in the data frame
        for ticker in self.tickers:
            # Get stock data
            temp = yf.download(
                ticker, 
                start=self.start, end=self.end, 
                proxy=self.proxy
            )
            temp["tic"] = ticker
     
            # Reset the index to use integers as index instead of dates
            temp = temp.reset_index()
      
            # Convert the column names to standardized names
            temp.columns = [
                "date", 
                "open", "high", "low", "close", "adjcp", 
                "volume", 
                "tic",
            ]
            
            # Create day of the week column (monday = 0)
            temp["day"] = temp["date"].dt.dayofweek
        
            # Convert date to standard string format, easy to filter
            temp["date"] = temp.date.apply(
                lambda item: item.strftime("%Y-%m-%d")
            )
        
            # Use adjusted close price instead of close price
            # so drop the adjusted close price column
            temp = temp.drop(labels="close", axis=1)
            
            # Rearrange columns 
            temp = temp[[
                "date", "tic", 
                "open", "high", "low", "adjcp", 
                "volume",
            ]]
        
            # Sort items
            temp = temp.sort_values(by=["date", "tic"])
            temp = temp.reset_index(drop=True)

            # Append to data frame
            df = df.append(temp)
        
        # Drop missing data
        df = df.dropna()
        df = df.reset_index(drop=True)

        return df
