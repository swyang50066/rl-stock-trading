import  os

import  numpy       as  np
import  pandas      as  pd

from    stockstats      import  StockDataFrame      as  sdf

from    preproc.uploader        import  Uploader


# Headers
ADJUSTED_VALUE_HEADS = [
    "date", "tic", 
    "prccd", "ajexdi", "prcod", "prchd", "prcld", "cshtrd"
]
OHLCV_VALUE_HEADS = [
    "date", "tic",
    "adjcp", "ajexdi", "open", "high", "low", "volume"
]

# Variables
DEFAULT_TECHNICAL_INDICATOR_LIST = [
    "macd", "rsi_30", "cci_30", "dx_30"
]
DEFAULT_USER_DEFINED_FEATURES = {
}


def _from_adjusted_to_ohlcv(df):
    ''' Convert Adjusted values to OHLCV
    '''
    # Fetch adjusted items
    x = df.copy()[ADJUSTED_VALUE_HEADS]

    # Fill non-zero daily adjustment factor
    x["ajexdi"] = x["ajexdi"].apply(
        lambda index: 1 if index == 0 else index
    )

    # Convert values
    x["adjcp"] = x["prccd"] / x["ajexdi"]
    x["open"] = x["prcod"] / x["ajexdi"]
    x["high"] = x["prchd"] / x["ajexdi"]
    x["low"] = x["prcld"] / x["ajexdi"]
    x["volume"] = x["cshtrd"]

    # Sort items
    x = x[OHLCV_VALUE_HEADS]
    x = x.sort_values(by=["date", "tic"])
    x = x.reset_index(drop=True)    # reindexing

    return x


def _from_ohlcv_to_adjusted(df):
    ''' Convert OHLCV values to Adjusted values
    '''
    # Fetch OHLCV items
    x = df.copy()[OHLCV_VALUE_HEADS]

    # Fill non-zero daily adjustment factor
    x["ajexdi"] = x["ajexdi"].apply(
        lambda index: 1 if index == 0 else index
    )

    # Convert values
    x["prccd"] = x["adjcp"] * x["ajexdi"]
    x["prcod"] = x["open"] * x["ajexdi"]
    x["prchd"] = x["high"] * x["ajexdi"]
    x["prcld"] = x["low"] * x["ajexdi"]
    x["schtrd"] = x["volume"]

    # Sort items
    x = x[ADJUSTED_VALUE_HEADS]
    x = x.sort_values(by=["date", "tic"])
    x = x.reset_index(drop=True)    # reindexing

    return x

def add_user_defined_feature(df, features):
    ''' Append 'user-defined-feature' to the data frame
    '''
    # Fetch data frame
    x = df.copy()
   
    # ====> 
    '''
    x["daily_return"] = df.cloase.pct_change(1)
    '''
    # <====

    return x


def add_volatility_index(df):
    ''' Append 'Volatility Index (VIX)' to the data frame 
    '''
    # Fetch data frame 
    x = df.copy()

    # Load from Yahoo        
    vix = Uploader(
        tickers=["^VIX"],
        start_date=df.date.min(), 
        end_date=df.date.max(), 
    ).fetch()

    # Rearrange columns 
    vix = vix[["date", "adjcp"]]
    vix.columns = ["date", "vix"]

    # Merge and sort items
    x = x.merge(vix, on="date")
    x = x.sort_values(by=["date", "tic"])
    x = x.reset_index(drop=True)
        
    return x


def add_technical_indicator(df, indicators):
    ''' Calculate technical indicators and append it to the dataframe
    '''
    # Switch columns
    x = df.copy()
    
    # Fetch items
    stock = sdf.retype(x.copy())
    stock["close"] = stock["adjcp"]
    
    # Get a list of stock tickers
    tickers = stock.tic.unique()

    # Calculate technical indicators
    for indicator in indicators:
        container = pd.DataFrame()
        for ticker in tickers:
            # Get indicator value
            temp = stock[stock.tic == ticker][indicator]
            temp = pd.DataFrame(temp)

            # Fill container items
            temp["tic"] = ticker
            temp["date"] = x[x.tic == ticker]["date"].to_list()
            container = container.append(temp, ignore_index=True)
        
        # Merge technical indicator into the data frame    
        x = x.merge(
            container[["tic", "date", indicator]], 
            on=["tic", "date"],
            how="left"
        )
 
    # Sort items
    x = x.sort_values(by=["date", "tic"]) 
    x = x.reset_index(drop=True)

    return x


def get_turbulence_index(df, start=252):
    ''' Calculate financial turbulence index
        
        (Consider after a year, default: start=252)
    '''
    # can add other market assets
    df_price_pivot = df.pivot(index="date", columns="tic", values="adjcp")
    # use returns to calculate turbulence
    df_price_pivot = df_price_pivot.pct_change()

    unique_date = df.date.unique()
        
    # start after a year
    turbulence = [0] * start
    
    count = 0
    for i in range(start, len(unique_date)):
        # Get current price
        current_price = df_price_pivot[
            df_price_pivot.index == unique_date[i]
        ]
            
        # Use one year rolling window to calcualte covariance
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - 252])
        ]
            
        # Drop tickers which has number missing values more than the "oldest" ticker
        filtered_hist_price = hist_price.iloc[
            hist_price.isna().sum().min() :
        ].dropna(axis=1)

        cov_temp = filtered_hist_price.cov()
        current_temp = (
            current_price[[x for x in filtered_hist_price]] 
            - np.mean(filtered_hist_price, axis=0)
        )

        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
            current_temp.values.T
        )
            
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # Avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        
        turbulence.append(turbulence_temp)

    # Return a set of tubulence index
    turbulence = pd.DataFrame(
        {"date": df_price_pivot.index, "turbulence": turbulence}
    )
        
    return turbulence


def add_turbulence_index(df):
    ''' Calculate turbulence index and append it to the dataframe
        (The index is based on dow 30)
    '''
    # Fetch data frame
    x = df.copy()

    # Calculate turbulence index
    turbulence = get_turbulence_index(x)

    # Merge and sort items
    x = x.merge(turbulence, on="date")
    x = x.sort_values(['date','tic'])
    x = x.reset_index(drop=True)

    return x


class DatasetIOMethod(object):
    @staticmethod
    def load_yahoo_finance(tickers, start_date, end_date):
        ''' Load Yahoo finance dataset
        '''
        uploader = Uploader(tickers, start_date, end_date)

        return uploader.fetch()

    @staticmethod
    def load_csv(tickers, file_name, b_adjusted=False):
        ''' Return dataframe by loading csv dataset file (.csv)
        '''
        # Set path of data csv file
        file_path = (
            os.path.dirname(os.path.realpath(__file__))
            + "/" + file_name
        )

        # Load dataset
        df = pd.read_csv(file_path)
        if b_adjusted:
            df = _from_adjusted_to_ohlcv(df)

        # Filter denoted tickers
        if tickers:
            df= df[df["tic"].isin(tickers)]
            df = df.sort_values(by=["tic", "date"])
            df = df.reset_index(drop=True)

        return df 

    @staticmethod
    def save_csv(df, file_name):
        ''' Save dataframe
        '''
        # Set path of data csv file
        file_path = (
            os.path.dirname(os.path.realpath(__file__))
            + "/" + file_name
        )

        # Save dataframe as .csv file
        df.to_csv(file_path)

    @staticmethod
    def batch(df, start_date=20090101, end_date=20201231):
        ''' Return a batch data from whole dataframe,
            of which being between 'start_date' and 'end_date'
        '''
        # Extract batch and sort items in order of 'date'
        df_batch = df[(df.datadata >= start_date) & (df.date < end_date)]
        df_batch = df_batch.sort_values(by=["date", "tic"])
        df_batch = df_batch.reset_index(drop=True)    # reindexing

        return df_batch


class Preprocessor(DatasetIOMethod):
    def __init__(self, technical_indicator_list=list(),
                       user_defined_features=dict(), 
                       b_use_technical_indicator=True,
                       b_use_volatility_index=True,
                       b_use_turbulence_index=True,
                       b_use_user_defined_index=True 
                ):
        super(Preprocessor, self).__init__()

        # Declare parameters
        if (isinstance(technical_indicator_list, list) and 
            technical_indicator_list):
            self.technical_indicator_list = technical_indicator_list
        else:
            self.technical_indicator_list = DEFAULT_TECHNICAL_INDICATOR_LIST
        if (isinstance(user_defined_features, dict) and
            user_defined_features):
            self.user_defined_features = user_defined_features
        else:
            self.user_defined_features = DEFAULT_USER_DEFINED_FEATURES

        self.b_use_technical_indicator = b_use_technical_indicator
        self.b_use_volatility_index = b_use_volatility_index
        self.b_use_turbulence_index = b_use_turbulence_index
        self.b_use_user_defined_index = b_use_user_defined_index

    def apply(self, df):
        ''' Apply preprocessing
        '''
        # Add technical indicators using stockstats
        if self.b_use_technical_indicator:
            print("Append technical indicators")
            df = add_technical_indicator(
                df, indicators=self.technical_indicator_list
            )

        # Add VIX for multiple stock
        if self.b_use_volatility_index:
            print("Append volatility index (VIX)")
            df = add_volatility_index(df)

        # Add turbulence index for multiple stock
        if self.b_use_turbulence_index:
            print("Append turbulence index")
            df = add_turbulence_index(df)
        
        # Add user defined feature
        print("Append user-defined-feature")
        if self.b_use_user_defined_index:
            print("Append user-defined-feature")
            df = add_user_defined_feature(
                df, features=self.user_defined_features
            )
        
        # Fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        df = df.sort_values(by=["date", "tic"])
        df = df.reset_index(drop=True) 
        
        return df
        
