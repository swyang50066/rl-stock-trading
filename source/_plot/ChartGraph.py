import  os
import  sys
import  time
import  datetime

import  numpy               as  np

import  pandas              as  pd
import  pandas_datareader   as  wb

import  matplotlib.pyplot   as  plt
import  matplotlib.ticker   as  ticker
import  matplotlib.gridspec as  gridspec

from    matplotlib.widgets                  import  RectangleSelector, Cursor, MultiCursor
from    matplotlib.backends.backend_qt5agg  import  FigureCanvasQTAgg       as  FigureCanvas
from    matplotlib.backends.backend_qt5agg  import  NavigationToolbar2QT    as  NavigationToolbar

from    mpl_finance         import  candlestick2_ohlc

class PlotStockChart(object):
    def DailyChart(self):
        self.fig.clf()
       
        code  = self.EditCodeLine.text()
        start = datetime.datetime(2018, 1, 1)
        end   = datetime.datetime(2018, 12, 31)

        df = wb.DataReader(code + '.ks', 'yahoo', start, end)
        df = df[df['Volume'] > 0]
        self.index = df.index.astype('str')
        
        df['MA5'] = df['Adj Close'].rolling(window=5, center=True, min_periods=1).mean()
        df['MA20'] = df['Adj Close'].rolling(window=20, center=True, min_periods=1).mean()
        df['MA60'] = df['Adj Close'].rolling(window=60, center=True, min_periods=1).mean()
        df['MA120'] = df['Adj Close'].rolling(window=120, center=True, min_periods=1).mean()
       
        gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1], sharex=ax1)
        ax1.get_xaxis().set_visible(False)
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(self.xdate))
        ax2.xaxis.set_major_locator(ticker.AutoLocator())
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(self.xdate))

        ax1.plot(self.index, df['MA5'], linewidth=1, label='MA20')
        ax1.plot(self.index, df['MA20'], linewidth=1, label='MA20')
        ax1.plot(self.index, df['MA60'], linewidth=1, label='MA60')
        ax1.plot(self.index, df['MA120'], linewidth=1, label='MA120')
        candlestick2_ohlc(ax1, df['Open'], df['High'], df['Low'], df['Close'], 
                          width=0.5, colorup='r', colordown='b') 
        ax2.bar(self.index, df['Volume'], color='Gray')
        
        self.RS = RectangleSelector(ax1, self.line_select_callback,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        
        self.MCS = MultiCursor(self.canvas, (ax1, ax2), color='r', lw=1) 
        self.CSH = Cursor(ax1, useblit=True, color='r', lw=1, horizOn=True, vertOn=True) 
        
        ax1.legend(loc='upper right')
        self.fig.tight_layout() 
        self.fig.autofmt_xdate()
        
        self.canvas.mpl_connect('key_press_event', self.toggle_selector)
        self.canvas.draw()

    def xdate(self, x, pos):
        try:
            return self.index[int(x-0.5)][:10]
        except IndexError:
            return ''

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

    def toggle_selector(self, event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.RS.active:
            print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)
