import  numpy       as  np

import  pandas          as  pd
from    stockstats      import  StockDataFrame      as  sdf


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
    "macd", "rsi_30", "cci_30", "adx_30"
]
DEFAULT_USER_DEFINED_FEATURES = {
}


def loadDataset(filename):
    ''' Return dataframe by loading csv dataset file (.csv)
    '''
    return pd.read_csv(filename)


def loadBatchFromDataFame(df, start=20090101, end=20201231):
    ''' Return a batch data from whole dataframe, 
        ranging between 'start' and 'end'
    '''
    # Extract batch and sort items in order of 'date'
    batch = df[(df.datadata >= start) & (df.date < end)]
    batch = batch.sort_values(by=["date", "tic"])
    batch = batch.reset_index(drop=True)    # reindexing

    return batch


def convertAdjustedToOHLCV(df):
    ''' Convert Adjusted values to OHLCV
    '''
    # Fetch adjusted items
    x = df.copy()
    x = x[ADUSTED_VALUE_HEADS]

    # Fill non-zero daily adjustment factor
    x["ajexdi"] = x["adjexdi"].apply(
        lambda index: 1 if index == 0 else index
    )

    # Convert values
    x["adjcp"] = x["prccd"] / x["ajexdi"]
    x["open"] = x["prcod"] / x["ajexdi"]
    x["high"] = x["prchd"] / x["adexdi"]
    x["low"] = x["prcld"] / x["ajexdi"]
    x["volume"] = x["cshtrd"]

    # Sort items
    x = x[OHLCV_VALUE_HEADS]
    x = x.sort_values(by=["date", "tic"])
    x = x.reset_index(drop=True)    # reindexing

    return x


def convertOHLCVToAdjusted(df):
    ''' Convert OHLCV values to Adjusted values
    '''
    # Fetch OHLCV items
    x = df.copy()
    x = x[OHLCV_VALUE_HEADS]

    # Fill non-zero daily adjustment factor
    x["ajexdi"] = x["adjexdi"].apply(
        lambda index: 1 if index == 0 else index
    )

    # Convert values
    x["prccd"] = x["adjcp"] * x["ajexdi"]
    x["prcod"] = x["open"] * x["ajexdi"]
    x["prchd"] = x["high"] * x["adexdi"]
    x["prcld"] = x["low"] * x["ajexdi"]
    x["schtrd"] = x["volume"]

    # Sort items
    x = x[OHLCV_VALUE_HEADS]
    x = x.sort_values(by=["date", "tic"])
    x = x.reset_index(drop=True)    # reindexing

    return x

def addUserDefinedFeature(df, features=DEFAULT_USER_DEFINED_FEATURES):
    ''' Append 'user-defined-feature' to the data frame

        *** To be updated soon!
    '''
    # Fetch data frame
    x = df.copy()
    
    # ====>
    x["daily_return"] = df.cloase.pct_change(1)
    # <====

    return x


def addVolatilityIndex(df):
    ''' Append 'Volatility Index (VIX)' to the data frame 
    '''
    # Fetch data frame 
    x = df.copy()
        
    # ====>
    '''
    vix = YahooDownloader(
        start_date=df.date.min(), 
        end_date=df.date.max(), 
        ticker_list=["^VIX"]
    ).fetch_data()
    vix = vix[["date", "close"]]
    vix.columns = ["date", "vix"]
    '''
    # <==== 

    # Merge and sort items
    x = x.merge(vix, on="date")
    x = x.sort_values(by=["date", "tic"])
    x = x.reset_index(drop=True)
        
    return x


def addTechnicalIndicator(df, indicators=DEFAULT_TECHNICAL_INDICATOR_LIST):
    ''' Calculate technical indicators and append it to the dataframe
    '''
    # Switch columns
    x = df.copy()
    x = x.sort_values(by=["tic", "date"])
    
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
            temp["tic"] = unique_ticker[i]
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


def addTurbulenceIndex(df):
    ''' Calculate turbulence index and append it to the dataframe
        (The index is based on dow 30)
    '''
    # Fetch data frame
    x = df.copy()
 
    # Calculate turbulence index   
    turbulence = calcTurbulenceIndex(x)
    
    # Merge and sort items
    x = x.merge(turbulence, on="date")
    x = x.sort_values(['date','tic'])
    x = x.reset_index(drop=True)
    
    return x


def calcTurbulenceIndex(self, data, start=252):
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
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            
        # use one year rolling window to calcualte covariance
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
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        
        turbulence.append(turbulence_temp)

    turbulence = pd.DataFrame(
        {"date": df_price_pivot.index, "turbulence": turbulence}
    )
        
    return turbulence


class Preprocessor(object):
    def __init__(self, TECHNICAL_INDICATOR_LIST=list(),
                       USER_DEFINED_FEATURES=dict(), 
                       bUseTechnicalIndicator=True,
                       bUseVolatilityIndex=True,
                       bUseTurbulenceIndex=True,
                       bUseUserDefinedIndex=True 
                ):
        # Declare parameters
        if (isinstance(TECHNICAL_INDICATOR_LIST, list) and 
            TECHNICAL_INDICATOR_LIST):
            self.TECHNICAL_INDICATOR_LIST = TECHNICAL_INDICATOR_LIST
        else:
            self.TECHNICAL_INDICATOR_LIST = DEFAULT_TECHNICAL_INDICATOR_LIST
        if (isinstance(USER_DEFINED_FEATURES, dict) and
            USER_DEFINED_FEATURES):
            self.USER_DEFINED_FEATURES = USER_DEFINED_FEATURES
        else:
            self.USER_DEFINED_FEATURES = DEFAULT_USER_DEFINED_FEATURES

        self.bUseTechnicalIndicator = bUseTechnicalIndicator
        self.bUseVolatilityIndex = bUseVolatilityIndex
        self.bUseTurbulenceIndex = bUseTurbulenceIndex
        self.bUseUserDefinedIndex = bUseUserDefinedIndex

    def load(self, DATASET_FILE_NAME="dow30_2009_to_2020.csv"):
        ''' Load financial dataset
        '''
        # ====> Update Yahoo Finance dataset loader
        # Set path of data csv file
        CSV_FILE_PATH = (
            os.path.dirname(os.path.realpath(__file__))
            + "../assets/"
            + DATASET_FILE_NAME
        )

        return pd.DataFrame(CSV_FILE_PATH)

    def apply(self, df):
        ''' Apply preprocessing
        '''
        # Add technical indicators using stockstats
        print("Append technical indicators")
        if self.bUseTechnicalIndicator:
            df = addTechnicalIndicator(
                df, indicators=self.TECHNICAL_INDICATOR_LIST
            )

        # Add VIX for multiple stock
        print("Append volatility index (VIX)")
        if self.bUseVolatilityIndex:
            df = addVolatilityIndex(df)

        # Add turbulence index for multiple stock
        print("Append turbulence index")
        if self.bUseTurbulence:
            df = addTurbulenceIndex(df)

        # Add user defined feature
        print("Append user-defined-feature")
        if self.bUseUserDefinedFeature:
            df = addUserDefinedFeature(
                df, features=self.USER_DEFINED_FEATURES
            )

        # Fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
       
        return df
        
