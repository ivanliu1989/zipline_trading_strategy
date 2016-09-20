# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:57:33 2016

@author: sky_x
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
from zipline.utils import tradingcalendar
import pytz


def initialize(context):
    # Quantopian backtester specific variables
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
    context.y = symbol('USO')
    context.X = symbol('GLD')
    # set_benchmark(context.y)
    
    
    # strategy specific variables
    context.lookback = 20 # used for regression
    context.z_window = 20 # used for zscore calculation, must be <= lookback
    
    context.useHRlag = True
    context.HRlag = 2
    
    context.spread = np.array([])
    context.hedgeRatioTS = np.array([])
    context.inLong = False
    context.inShort = False
    context.entryZ = 1.0
    context.exitZ = 0.0

    if not context.useHRlag:
        # a lag of 1 means no-lag, this is used for np.array[-1] indexing
        context.HRlag = 1
        
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
   
    _USO_value = context.portfolio.positions[symbol('USO')].amount * context.portfolio.positions[symbol('USO')].last_sale_price
    _GLD_value = context.portfolio.positions[symbol('GLD')].amount * context.portfolio.positions[symbol('GLD')].last_sale_price
    _leverage = (abs(_USO_value) + abs(_GLD_value)) / context.portfolio.portfolio_value
    record(
            GLD_value = _GLD_value ,
            USO_value = _USO_value ,
            leverage = _leverage
    )
    
    if get_open_orders():
        return
    
    now = get_datetime()
    exchange_time = now.astimezone(pytz.timezone('US/Eastern'))
    
    if not (exchange_time.hour == 15 and exchange_time.minute == 30):
        return
    
    prices = history(35, '1d', 'price').iloc[-context.lookback::]

    y = prices[context.y]
    X = prices[context.X]

    try:
        hedge = hedge_ratio(y, X, add_const=True)      
    except ValueError as e:
        log.debug(e)
        return
    
    context.hedgeRatioTS = np.append(context.hedgeRatioTS, hedge)
    # Calculate the current day's spread and add it to the running tally
    if context.hedgeRatioTS.size < context.HRlag:
        return
    # Grab the previous day's hedgeRatio
    hedge = context.hedgeRatioTS[-context.HRlag]  
    context.spread = np.append(context.spread, y[-1] - hedge * X[-1])

    if context.spread.size > context.z_window:
        # Keep only the z-score lookback period
        spreads = context.spread[-context.z_window:]
        
        zscore = (spreads[-1] - spreads.mean()) / spreads.std()
          
        if context.inShort and zscore < 0.0:
            order_target(context.y, 0)
            order_target(context.X, 0)
            context.inShort = False
            context.inLong = False
            record(USO_pct=0, GLD_pct=0)
            return
        
        if context.inLong and zscore > 0.0:
            order_target(context.y, 0)
            order_target(context.X, 0)
            context.inShort = False
            context.inLong = False
            record(USO_pct=0, GLD_pct=0)
            return
            
        if zscore < -1.0 and (not context.inLong):
            # Only trade if NOT already in a trade
            y_target_shares = 1
            X_target_shares = -hedge
            context.inLong = True
            context.inShort = False
            
            (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares,X_target_shares, y[-1], X[-1] )
            order_target_percent(context.y, y_target_pct)
            order_target_percent(context.X, x_target_pct)
            record(USO_pct=y_target_pct, GLD_pct=x_target_pct)
            return

        if zscore > 1.0 and (not context.inShort):
            # Only trade if NOT already in a trade
            y_target_shares = -1
            X_target_shares = hedge
            context.inShort = True
            context.inLong = False
           
            (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, y[-1], X[-1] )
            order_target_percent(context.y, y_target_pct)
            order_target_percent(context.X, x_target_pct)
            record(USO_pct=y_target_pct, GLD_pct=x_target_pct)
        
#        record(
            # Z=zscore ,
            # hedge_ratio = hedge ,
            # spread=context.spread[-1] ,
            # portfolioValue = context.portfolio.portfolio_value ,
#            GLD_value = _GLD_value ,
#            USO_value = _USO_value ,
#            leverage = _leverage
#        )


def is_market_close(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    return dt == tradingcalendar.open_and_closes.T[ref]['market_close']

def hedge_ratio(y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.params[1]
    model = sm.OLS(y, X).fit()
    return model.params.values
    
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)