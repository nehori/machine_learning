import talib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# グラフ出力
def display_chart(data, x):
    sns.set(font_scale = 1.2)
    fig = plt.figure() 
    for i in range(len(data)):
        plt.plot(x, data[i][0], label=data[i][1])
    plt.xticks(range(0, len(x), 100), x[::100])
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()    
    # ファイルに保存
    fig.savefig("img.png")

def vr(df, window=26, type=1):
    """
    Volume Ratio (VR)
    Formula:
    VR[A] = SUM(av + cv/2, n) / SUM(bv + cv/2, n) * 100
    VR[B] = SUM(av + cv/2, n) / SUM(av + bv + cv, n) * 100
    Wako VR = SUM(av - bv - cv, n) / SUM(av + bv + cv, n) * 100
        av = volume if close > pre_close else 0
        bv = volume if close < pre_close else 0
        cv = volume if close = pre_close else 0
    """
    df['av'] = np.where(df['close'].diff() > 0, df['volume'], 0)
    avs = df['av'].rolling(window=window, center=False).sum()
    df['bv'] = np.where(df['close'].diff() < 0, df['volume'], 0)
    bvs = df['bv'].rolling(window=window, center=False).sum()
    df['cv'] = np.where(df['close'].diff() == 0, df['volume'], 0)
    cvs = df['cv'].rolling(window=window, center=False).sum()
    df.drop(['av', 'bv', 'cv'], inplace=True, axis=1)
    if type == 1: # VR[A]
       vr = (avs + cvs / 2) / (bvs + cvs / 2) * 100  
    elif type == 2: # VR[B]
       vr = (avs + cvs / 2) / (avs + bvs + cvs) * 100
    else: # Wako VR
       vr = (avs - bvs - cvs) / (avs + bvs + cvs) * 100
    return vr

# Bollinger Bands  
def BBANDS(df, n=25):  
    MA = pd.Series(pd.rolling_mean(df['Close'], n))  
    MSD = pd.Series(pd.rolling_std(df['Close'], n))  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
    df = df.join(B2)  
    return df

# MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

def add_talib_features(df):
    # 単純移動平均(Simple Moving Average)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    df['sma3'] = talib.SMA(close, timeperiod=3)
    df['sma15'] = talib.SMA(close, timeperiod=15)
    df['sma50'] = talib.SMA(close, timeperiod=50)
    df['sma75'] = talib.SMA(close, timeperiod=75)
    df['sma100'] = talib.SMA(close, timeperiod=100)
    # ボリンジャーバンド(Bollinger Bands)
    df['upper1'], middle, df['lower1'] = talib.BBANDS(close, timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
    df['upper2'], middle, df['lower2'] = talib.BBANDS(close, timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)
    df['upper3'], middle, df['lower3'] = talib.BBANDS(close, timeperiod=25, nbdevup=3, nbdevdn=3, matype=0)
    # MACD - Moving Average Convergence/Divergence
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # RSI - Relative Strength Index
    df['rsi9'] = talib.RSI(close, timeperiod=9)
    df['rsi14'] = talib.RSI(close, timeperiod=14)
    # 平均方向性指数（ADX）
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    # ATR(Average True Range)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    # 移動平均乖離率(MA Deviation)
    sma5 = talib.SMA(close, timeperiod=5)
    sma15 = talib.SMA(close, timeperiod=15)
    sma25 = talib.SMA(close, timeperiod=25)
    df['diffma5'] = 100 * (close - sma5) / sma5
    df['diffma15'] = 100 * (close - sma15) / sma15
    df['diffma25'] = 100 * (close - sma25) / sma25
    # 前日比
    df['diff1'] = close / df['close'].shift(1)
    df['diff2'] = df['close'].shift(1) / df['close'].shift(2)
    df['diff3'] = df['close'].shift(2) / df['close'].shift(3)
    return df

# テクニカル指標
def add_new_features(df):
    # Exception: inputs are all NaN 回避
    if not df['close'].isnull().all():
       df = add_talib_features(df)
    # VR - Volume Ratio
    df['vr'] = vr(df, 25)
    # closeの欠損値が含まれている行を削除
    df = df.dropna(subset=["close"])
    # closeの欠損値が含まれている行を削除
    df = df.dropna(subset=["close"])
    return df

# Protraファイルの作成
def create_protra_dataset(code, date, y_pred, flag):
    # 利益が高いと判定したものだけ残す
    y_pred = np.where(y_pred >= flag, True, False)
    s = ""
    s += "  if ((int)Code == " + code + ")\n"
    s += "     if ( \\\n"
    for i in range(len(y_pred)):
        if(y_pred[i]):
             (year, month, day) = date[i].split('/')
             s += "(Year == " + str(int(year)) + " && Month == " + str(int(month)) + " && Day == " + str(int(day)) + ") || \\\n"
    s += "         (Year == 3000))\n"
    s += "         return 1\n"
    s += "     end\n"
    s += "  end\n"
    return s

def merge_protra_dataset(s):
    a = "def IsBUYDATE\n"
    b = "   return 0\n"
    b += "end\n"
    return a + s + b
