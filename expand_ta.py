

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
def expand_length_array(data, row_ta, attri, func, fillna=0):
    if attri in row_ta:
        if not row_ta[attri]==0 and not row_ta[attri]=='0':
            arr = str(row_ta[attri]).split(';')
            #arr = [5,10,15,20,25,30,40,45,50,60,80,90,100,120]   #临时方案
            #arr = [5,10,15,30,40,60,120,180]
            for vvv in arr:
                vvv = int(vvv)
                func(length=vvv, append=True, fillna=fillna)

# fast:slow
def expand_fast_slow(data, row_ta, attri, func):
    if attri in row_ta:
        if not row_ta[attri]==0 and not row_ta[attri]=='0':
            arr = str(row_ta[attri]).split(':')
            fast = int(arr[0])
            slow = int(arr[1])
            func(fast=fast, slow=slow, append=True, fillna=0)

# fast:slow:length
def expand_fast_slow_length(data, row_ta, attri, func):
    if attri in row_ta:
        if not row_ta[attri]==0 and not row_ta[attri]=='0':
            arr = str(row_ta[attri]).split(':')
            fast = int(arr[0])
            slow = int(arr[1])
            length = int(arr[2])
            func(fast=fast, slow=slow, length=length, append=True, fillna=0)


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

    print("1111111")
    
    #data.ta.log_return(cumulative=True, append=True)
    #data.ta.percent_return(cumulative=True, append=True)

    expand_length_array(data, row_ta, 'aberration', data.ta.aberration)       #1
    #expand_simple(data, row_ta, 'above', data.ta.above)                      #2
    #expand_simple(data, row_ta, 'above_value', data.ta.above_value)          #3
    expand_length_array(data, row_ta, 'accbands', data.ta.accbands)           #4
    expand_simple(data, row_ta, 'ad', data.ta.ad)                             #5
    expand_simple(data, row_ta, 'adosc', data.ta.adosc)                       #6
    expand_length_array(data, row_ta, 'adx', data.ta.adx)                     #7
    expand_length_array(data, row_ta, 'alma', data.ta.alma)                   #8
    expand_fast_slow(data, row_ta, 'amat', data.ta.amat)                      #9
    expand_fast_slow(data, row_ta, 'ao', data.ta.ao)                          #10
    expand_fast_slow(data, row_ta, 'aobv', data.ta.aobv)                      #11
    expand_fast_slow(data, row_ta, 'apo', data.ta.apo)                        #12
    expand_length_array(data, row_ta, 'aroon', data.ta.aroon)                 #13
    expand_length_array(data, row_ta, 'atr', data.ta.atr)                     #14
    expand_length_array(data, row_ta, 'bbands', data.ta.bbands)               #15
    #expand_simple(data, row_ta, 'below', data.ta.below)                      #16



    #expand_simple(data, row_ta, 'below_value', data.ta.below_value)          #17
    expand_length_array(data, row_ta, 'bias', data.ta.bias)                   #18
    expand_simple(data, row_ta, 'bop', data.ta.bop)                           #19
    expand_length_array(data, row_ta, 'brar', data.ta.brar)                   #20
    expand_length_array(data, row_ta, 'cci', data.ta.cci)                     #21
    expand_simple(data, row_ta, 'cdl_pattern', data.ta.cdl_pattern)           #22
    expand_simple(data, row_ta, 'cdl_z', data.ta.cdl_z)                       #23
    expand_length_array(data, row_ta, 'cfo', data.ta.cfo)                     #24
    expand_length_array(data, row_ta, 'cg', data.ta.cg)                       #25
    expand_length_array(data, row_ta, 'chop', data.ta.chop)                   #26
    expand_simple(data, row_ta, 'cksp', data.ta.cksp)                         #27
    expand_length_array(data, row_ta, 'cmf', data.ta.cmf)                     #28
    expand_length_array(data, row_ta, 'cmo', data.ta.cmo)                     #29
    expand_length_array(data, row_ta, 'coppock', data.ta.coppock)             #30        这个比较特殊，既有length，又有fast，slow
    #expand_simple(data, row_ta, 'cross', data.ta.cross)                      #31
    expand_simple(data, row_ta, 'cross_value', data.ta.cross_value)           #32


    #expand_length_array(data, row_ta, 'cti', data.ta.cti, fillna=1)          #33
    expand_length_array(data, row_ta, 'decay', data.ta.decay)                 #34
    expand_length_array(data, row_ta, 'decreasing', data.ta.decreasing)       #35
    expand_length_array(data, row_ta, 'dema', data.ta.dema)                   #36
    expand_simple(data, row_ta, 'dm', data.ta.dm)                             #37
    expand_simple(data, row_ta, 'donchian', data.ta.donchian)                 #38
    expand_length_array(data, row_ta, 'dpo', data.ta.dpo)                     #39
    expand_length_array(data, row_ta, 'ebsw', data.ta.ebsw)                   #40
    expand_length_array(data, row_ta, 'efi', data.ta.efi)                     #41
    expand_length_array(data, row_ta, 'ema', data.ta.ema)                     #42
    expand_length_array(data, row_ta, 'entropy', data.ta.entropy)             #43
    expand_length_array(data, row_ta, 'eom', data.ta.eom)                     #44
    expand_length_array(data, row_ta, 'er', data.ta.er)                       #45
    expand_length_array(data, row_ta, 'eri', data.ta.eri)                     #46
    expand_length_array(data, row_ta, 'fisher', data.ta.fisher)               #47
    expand_length_array(data, row_ta, 'fwma', data.ta.fwma)                   #48
    expand_simple(data, row_ta, 'ha', data.ta.ha)                             #49
    expand_simple(data, row_ta, 'hilo', data.ta.hilo)                         #50
    expand_simple(data, row_ta, 'hl2', data.ta.hl2)                           #51



    # log_return,

    expand_simple(data, row_ta, 'hlc3', data.ta.hlc3)                         #52
    expand_length_array(data, row_ta, 'hma', data.ta.hma)                     #53
    expand_simple(data, row_ta, 'hwc', data.ta.hwc)                           #54
    expand_simple(data, row_ta, 'hwma', data.ta.hwma)                         #55
    #expand_simple(data, row_ta, 'ichimoku', data.ta.ichimoku)                #56
    expand_length_array(data, row_ta, 'increasing', data.ta.increasing)       #57
    expand_length_array(data, row_ta, 'inertia', data.ta.inertia)             #58
    expand_length_array(data, row_ta, 'jma', data.ta.jma)                     #59
    expand_length_array(data, row_ta, 'kama', data.ta.kama)                   #60       这个比较特殊，既有length，又有fast，slow
    expand_length_array(data, row_ta, 'kc', data.ta.kc)                       #61
    expand_length_array(data, row_ta, 'kdj', data.ta.kdj)                     #62
    expand_simple(data, row_ta, 'kst', data.ta.kst)                           #63
    expand_length_array(data, row_ta, 'kurtosis', data.ta.kurtosis)           #64
    expand_fast_slow(data, row_ta, 'kvo', data.ta.kvo)                        #65
    expand_length_array(data, row_ta, 'linreg', data.ta.linreg)               #66
    expand_simple(data, row_ta, 'log_return', data.ta.log_return)             #67



    #expand_fast_slow_length(data, row_ta, 'long_run', data.ta.long_run)       #68       这个比较特殊，既有length，又有fast，slow
    expand_fast_slow(data, row_ta, 'macd', data.ta.macd)                      #69
    expand_length_array(data, row_ta, 'mad', data.ta.mad)                     #70
    expand_fast_slow(data, row_ta, 'massi', data.ta.massi)                    #71
    expand_length_array(data, row_ta, 'mcgd', data.ta.mcgd)                   #72
    expand_length_array(data, row_ta, 'median', data.ta.median)               #73
    expand_length_array(data, row_ta, 'mfi', data.ta.mfi)                     #74
    expand_length_array(data, row_ta, 'midpoint', data.ta.midpoint)           #75
    expand_length_array(data, row_ta, 'midprice', data.ta.midprice)           #76
    expand_length_array(data, row_ta, 'mom', data.ta.mom)                     #77
    expand_length_array(data, row_ta, 'natr', data.ta.natr)                   #78
    expand_length_array(data, row_ta, 'nvi', data.ta.nvi)                     #79
    expand_simple(data, row_ta, 'obv', data.ta.obv)                           #80
    expand_simple(data, row_ta, 'ohlc4', data.ta.ohlc4)                       #81
    expand_simple(data, row_ta, 'pdist', data.ta.pdist)                       #82
    expand_simple(data, row_ta, 'percent_return', data.ta.percent_return)     #83


    expand_length_array(data, row_ta, 'pgo', data.ta.pgo)                     #84
    expand_fast_slow(data, row_ta, 'ppo', data.ta.ppo)                        #85
    expand_simple(data, row_ta, 'psar', data.ta.psar)                         #86
    expand_length_array(data, row_ta, 'psl', data.ta.psl)                     #87
    expand_length_array(data, row_ta, 'pvi', data.ta.pvi)                     #88
    expand_fast_slow(data, row_ta, 'pvo', data.ta.pvo)                        #89
    expand_simple(data, row_ta, 'pvol', data.ta.pvol)                         #90
    expand_simple(data, row_ta, 'pvr', data.ta.pvr)                           #91
    expand_simple(data, row_ta, 'pvt', data.ta.pvt)                           #92
    expand_length_array(data, row_ta, 'pwma', data.ta.pwma)                   #93
    expand_length_array(data, row_ta, 'qqe', data.ta.qqe)                     #94
    expand_length_array(data, row_ta, 'qstick', data.ta.qstick)               #95
    expand_length_array(data, row_ta, 'quantile', data.ta.quantile)           #96
    expand_length_array(data, row_ta, 'rma', data.ta.rma)                     #97
    expand_length_array(data, row_ta, 'roc', data.ta.roc)                     #98
    expand_length_array(data, row_ta, 'rsi', data.ta.rsi)                     #99
    expand_length_array(data, row_ta, 'rsx', data.ta.rsx)                     #100
    expand_length_array(data, row_ta, 'rvgi', data.ta.rvgi)                   #101
    expand_length_array(data, row_ta, 'rvi', data.ta.rvi)                     #102
    #expand_fast_slow(data, row_ta, 'short_run', data.ta.short_run)           #103


    expand_length_array(data, row_ta, 'sinwma', data.ta.sinwma)               #104
    expand_length_array(data, row_ta, 'skew', data.ta.skew)                   #105
    expand_length_array(data, row_ta, 'slope', data.ta.slope)                 #106
    expand_length_array(data, row_ta, 'sma', data.ta.sma)                     #107
    expand_fast_slow(data, row_ta, 'smi', data.ta.smi)                        #108
    expand_simple(data, row_ta, 'squeeze', data.ta.squeeze)                   #109
    expand_simple(data, row_ta, 'squeeze_pro', data.ta.squeeze_pro)           #110
    expand_length_array(data, row_ta, 'ssf', data.ta.ssf)                     #111
    expand_simple(data, row_ta, 'stc', data.ta.stc)                           #112
    expand_length_array(data, row_ta, 'stdev', data.ta.stdev)                 #113
    expand_simple(data, row_ta, 'stoch', data.ta.stoch)                       #114
    expand_length_array(data, row_ta, 'stochrsi', data.ta.stochrsi)           #115
    expand_length_array(data, row_ta, 'supertrend', data.ta.supertrend)       #116
    expand_length_array(data, row_ta, 'swma', data.ta.swma)                   #117
    expand_length_array(data, row_ta, 't3', data.ta.t3)                       #118
    expand_simple(data, row_ta, 'td_seq', data.ta.td_seq)                     #119
    expand_length_array(data, row_ta, 'tema', data.ta.tema)                   #120



    expand_simple(data, row_ta, 'thermo', data.ta.thermo)                     #121
    #expand_length_array(data, row_ta, 'tos_stdevall', data.ta.tos_stdevall)   #122
    expand_length_array(data, row_ta, 'trima', data.ta.trima)                 #123
    expand_length_array(data, row_ta, 'trix', data.ta.trix)                   #124
    expand_simple(data, row_ta, 'true_range', data.ta.true_range)             #125
    expand_fast_slow(data, row_ta, 'tsi', data.ta.tsi)                        #126
    #expand_simple(data, row_ta, 'tsignals', data.ta.tsignals)                 #127
    expand_length_array(data, row_ta, 'ttm_trend', data.ta.ttm_trend)         #128
    expand_length_array(data, row_ta, 'ui', data.ta.ui)                       #129
    expand_simple(data, row_ta, 'uo', data.ta.uo)                             #130
    expand_length_array(data, row_ta, 'variance', data.ta.variance)           #131
    expand_length_array(data, row_ta, 'vhf', data.ta.vhf)                     #132
    expand_length_array(data, row_ta, 'vidya', data.ta.vidya)                 #133
    expand_simple(data, row_ta, 'vortex', data.ta.vortex)                     #134
    #expand_simple(data, row_ta, 'vp', data.ta.vp)                             #135
    #expand_simple(data, row_ta, 'vwap', data.ta.vwap)                         #136



    expand_length_array(data, row_ta, 'vwma', data.ta.vwma)                   #137
    expand_simple(data, row_ta, 'wcp', data.ta.wcp)                           #138
    expand_length_array(data, row_ta, 'willr', data.ta.willr)                 #139
    expand_length_array(data, row_ta, 'wma', data.ta.wma)                     #140
    #expand_simple(data, row_ta, 'xsignals', data.ta.xsignals)                 #141
    expand_length_array(data, row_ta, 'zlma', data.ta.zlma)                   #142
    expand_length_array(data, row_ta, 'zscore', data.ta.zscore)               #143

    data.ta.zscore



    #expand_simple(data, row_ta, 'log_return', data.ta.log_return)
    #expand_simple(data, row_ta, 'percent_return', data.ta.percent_return)
    #expand_simple(data, row_ta, 'obv', data.ta.obv)
    #expand_simple(data, row_ta, 'psar', data.ta.psar)





    #expand_length_array(data, row_ta, 'sma', data.ta.sma)
    #expand_length_array(data, row_ta, 'ema', data.ta.ema)
    #expand_length_array(data, row_ta, 'rsi', data.ta.rsi)
    #expand_length_array(data, row_ta, 'kdj', data.ta.kdj)

    #expand_length_array(data, row_ta, 'bias', data.ta.bias)
    #expand_length_array(data, row_ta, 'adx', data.ta.adx)
    #expand_length_array(data, row_ta, 'cmo', data.ta.cmo)
    #expand_length_array(data, row_ta, 'cci', data.ta.cci)
    #expand_length_array(data, row_ta, 'trix', data.ta.trix)
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