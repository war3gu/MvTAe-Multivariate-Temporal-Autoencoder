

from defines import *

import math

import numpy as np
import pandas as pd


# 1
def expand_simple(data, row_ta, attri, func):
    if attri in row_ta:
        if not row_ta[attri]==0 and not row_ta[attri]=='0':
            func(cumulative=True, append=True)

# v1;v2;v3
def expand_int_array(data, row_ta, attri, func):
    if attri in row_ta:
        if not row_ta[attri]==0 and not row_ta[attri]=='0':
            arr = str(row_ta[attri]).split(';')
            for vvv in arr:
                vvv = int(vvv)
                func(length=vvv, append=True, fillna=0)

# fast:slow
def expand_fast_slow(data, row_ta, attri, func):
    if attri in row_ta:
        if not row_ta[attri]==0 and not row_ta[attri]=='0':
            arr = str(row_ta[attri]).split(':')
            fast = int(arr[0])
            slow = int(arr[1])
            func(fast=fast, slow=slow, append=True, fillna=0)


def expand_all(data, row_ta):
    '''
    aberration, above, above_value, accbands, ad, adosc, adx, alma, amat, ao, aobv, apo, aroon, atr, bbands, below,
below_value, bias, bop, brar, cci, cdl_pattern, cdl_z, cfo, cg, chop, cksp, cmf, cmo, coppock, cross, cross_value,
cti, decay, decreasing, dema, dm, donchian, dpo, ebsw, efi, ema, entropy, eom, er, eri, fisher, fwma, ha, hilo, hl2,
hlc3, hma, hwc, hwma, ichimoku, increasing, inertia, jma, kama, kc, kdj, kst, kurtosis, kvo, linreg, log_return,
long_run, macd, mad, massi, mcgd, median, mfi, midpoint, midprice, mom, natr, nvi, obv, ohlc4, pdist, percent_return,
pgo, ppo, psar, psl, pvi, pvo, pvol, pvr, pvt, pwma, qqe, qstick, quantile, rma, roc, rsi, rsx, rvgi, rvi, short_run,
sinwma, skew, slope, sma, smi, squeeze, squeeze_pro, ssf, stc, stdev, stoch, stochrsi, supertrend, swma, t3, td_seq, tema,
thermo, tos_stdevall, trima, trix, true_range, tsi, tsignals, ttm_trend, ui, uo, variance, vhf, vidya, vortex, vp, vwap,
vwma, wcp, willr, wma, xsignals, zlma, zscore
   '''

    print("xxxxx")

    expand_int_array(data, row_ta, 'aberration', data.ta.aberration)       #1
    #expand_simple(data, row_ta, 'above', data.ta.above)                   #2
    #expand_simple(data, row_ta, 'above_value', data.ta.above_value)       #3
    expand_int_array(data, row_ta, 'accbands', data.ta.accbands)           #4
    expand_simple(data, row_ta, 'ad', data.ta.ad)                          #5
    expand_simple(data, row_ta, 'adosc', data.ta.adosc)                    #6
    expand_int_array(data, row_ta, 'adx', data.ta.adx)                     #7
    expand_int_array(data, row_ta, 'alma', data.ta.alma)                   #8
    expand_fast_slow(data, row_ta, 'amat', data.ta.amat)                   #9
    expand_fast_slow(data, row_ta, 'ao', data.ta.ao)                       #10
    expand_fast_slow(data, row_ta, 'aobv', data.ta.aobv)                   #11
    expand_fast_slow(data, row_ta, 'apo', data.ta.apo)                     #12
    expand_int_array(data, row_ta, 'aroon', data.ta.aroon)                 #13
    expand_int_array(data, row_ta, 'atr', data.ta.atr)                     #14
    expand_int_array(data, row_ta, 'bbands', data.ta.bbands)               #15
    #expand_simple(data, row_ta, 'below', data.ta.below)                   #16

    #data.ta.below

    #expand_simple(data, row_ta, 'log_return', data.ta.log_return)
    #expand_simple(data, row_ta, 'percent_return', data.ta.percent_return)
    #expand_simple(data, row_ta, 'obv', data.ta.obv)
    #expand_simple(data, row_ta, 'psar', data.ta.psar)





    #expand_int_array(data, row_ta, 'sma', data.ta.sma)
    #expand_int_array(data, row_ta, 'ema', data.ta.ema)
    #expand_int_array(data, row_ta, 'rsi', data.ta.rsi)
    #expand_int_array(data, row_ta, 'kdj', data.ta.kdj)

    #expand_int_array(data, row_ta, 'bias', data.ta.bias)
    #expand_int_array(data, row_ta, 'adx', data.ta.adx)
    #expand_int_array(data, row_ta, 'cmo', data.ta.cmo)
    #expand_int_array(data, row_ta, 'cci', data.ta.cci)
    #expand_int_array(data, row_ta, 'trix', data.ta.trix)
    #expand_fast_slow(data, row_ta, 'macd', data.ta.macd)


def expand_data(data, row_ta):

    for key, value in row_ta.items():           #缺失值，补充0
        if not isinstance(value, str):
            if math.isnan(value):
                row_ta[key] = '0'

    #data.set_index(pd.DatetimeIndex(data["Index"]), inplace=True)
    # Calculate Returns and append to the df DataFrame
    data = data.rename(columns={1: "open", 2: "high", 3: "low", 4: "close", 5: "amount", 6: "volume"})

    if 7 in data.columns:
        data = data.drop([7], axis=1)

    #for i in features_x:
    #print(i)

    expand_all(data, row_ta)



    data = data.fillna(0)
    data = data.replace(np.inf, -1)

    macroFeature.features_x = list(data.columns.values)

    print("features_x  {0}".format(macroFeature.features_x))

    # New Columns with results
    #data.columns
    # Take a peek
    #data.tail()

    # Create a DataFrame so 'ta' can be used.
    df = pd.DataFrame()
    # Help about this, 'ta', extension
    #help(df.ta)
    # List of all indicators
    #df.ta.indicators()


    return data