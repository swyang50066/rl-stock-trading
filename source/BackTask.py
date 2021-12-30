import  os
import  sys
import  time
import  datetime

import  numpy               as  np

import  pandas              as  pd
import  pandas_datareader   as  wb

import  matplotlib.pyplot   as  plt
from    matplotlib.backends.backend_qt5agg  import  FigureCanvasQTAgg       as  FigureCanvas
from    matplotlib.backends.backend_qt5agg  import  NavigationToolbar2QT    as  NavigationToolbar

from    zipline.api         import  order, symbol
from    zipline.algorithm   import  TradingAlgorithm

from    PyQt5.QtWidgets     import  *
from    PyQt5               import  uic

class PlotBackTask(object):
    def PlotBackTask(self):
        code  = self.EditCodeLine.text()
        start = datetime.datetime(2014, 1, 1)
        end   = datetime.datetime(2016, 3, 19)
        df    = wb.DataReader("AAPL", "yahoo", start, end)

        df = df[['Adj Close']]
        df.columns = [code]
        df = df.tz_localize("UTC")

        df2 = wb.DataReader(code + ".ks", "yahoo", start, end)
        df2 = df2[['Adj Close']]
        df2.columns = [code]

        df = df[len(df) - len(df2):]  #데이터 프레임의 row 수를 맞추는 작업
        df2[code] = np.where(1, df[code], df[code])

        algo = TradingAlgorithm(initialize=self.initialize, handle_data=self.handle_data)
        result = algo.run(df)

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.plot(result.index, result.portfolio_value)
        ax.grid()

        self.canvas.draw()

    def initialize(self,context):
        pass

    def handle_data(self,context, data):
        order(symbol(self.EditCodeLine.text()), 1)
