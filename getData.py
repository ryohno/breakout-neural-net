import alpaca_trade_api as tradeapi
import pandas as pd
import argparse
import datetime
import os
import numpy as np
from pandas_ta import vwma
from pandas_ta import sma
from pandas_ta import rsi



global market_df


def generate_features_data():
    market_data_df = pd.DataFrame(columns=['timestamp', 'high', 'low', 'open', 'close', 'volume'])
    market_data_df = market_data_df.append(data())
    timestamp_df = market_data_df['timestamp'].copy(deep=True)
    slow_vwma_value_df = pd.DataFrame(vwma(market_data_df.close, market_data_df.volume, length=15, offset=0))
    fast_vwma_value_df = pd.DataFrame(vwma(market_data_df.close, market_data_df.volume, length=5, offset=0))
    slow_sma_value_df = pd.DataFrame(sma(market_data_df.close, 15))
    fast_sma_value_df = pd.DataFrame(sma(market_data_df.close, 5))
    rsi_data_value_df = pd.DataFrame(rsi(market_data_df.close, 10))

    slow_vwma_df = pd.concat([timestamp_df, slow_vwma_value_df], axis=1)
    fast_vwma_df = pd.concat([timestamp_df, fast_vwma_value_df], axis=1)
    slow_sma_df = pd.concat([timestamp_df, slow_sma_value_df], axis=1)
    fast_sma_df = pd.concat([timestamp_df, fast_sma_value_df], axis=1)
    rsi_df = pd.concat([timestamp_df, rsi_data_value_df], axis=1)

    slow_vwma_df.to_csv('slow_vwma_data.csv', index=False)
    fast_vwma_df.to_csv('fast_vwma_data.csv', index=False)
    slow_sma_df.to_csv('slow_sma_data.csv', index=False)
    fast_sma_df.to_csv('fast_sma_data.csv', index=False)
    rsi_df.to_csv('rsi_data.csv', index=False)

def data():
    global pre_mask
    global ah_mask
    global market_df
    big_df = pd.DataFrame(columns=['timestamp', 'high', 'low', 'open', 'close', 'volume'])
    converted_start_date = datetime.datetime.strptime(params[1], '%Y-%m-%d')
    converted_end_date = datetime.datetime.strptime(params[2], '%Y-%m-%d')
    one_day_delta = datetime.timedelta(days=1)
    while converted_start_date <= converted_end_date:
        working_end_date_before_conversion = converted_start_date
        working_end_date = working_end_date_before_conversion.strftime('%Y-%m-%d')
        working_start_date = converted_start_date.strftime('%Y-%m-%d')
        returned_df = main(params[0], working_start_date, working_end_date, params[3], params[4])
        big_df = big_df.append(returned_df)
        converted_start_date += one_day_delta
    pre_mask = big_df['timestamp'].dt.time < datetime.time(9, 0)
    ah_mask = big_df['timestamp'].dt.time > datetime.time(16, 0)
    market_df = big_df[(~pre_mask) & (~ah_mask)].copy()
    market_df.to_csv('june3_IWM.csv', index=False)
    return market_df

def main(symbol, start_date, end_date, timespan, multiplier):

    bars = tradeapi.REST().polygon.historic_agg_v2(symbol, multiplier, timespan, start_date, end_date)
    df = pd.DataFrame({
        'timestamp': [bar.timestamp for bar in bars],
        'high': [bar.high for bar in bars],
        'low': [bar.low for bar in bars],
        'open': [bar.open for bar in bars],
        'close': [bar.close for bar in bars],
        'volume': [bar.volume for bar in bars]
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', type=str, default='SPY', help='symbol you want to get data for')
    parser.add_argument('start_date', type=str, default='2020-03-01', help='start date you want to get data for')
    parser.add_argument('end_date', type=str, default='2020-05-01', help='end date you want to get data for')
    parser.add_argument('timespan', type=str, default='minute', help='minute, daily, etc')
    parser.add_argument('multiplier', type=int, default=1, help='multiplier of the timespan')
    arg = parser.parse_args()
    global params
    params = [arg.symbol, arg.start_date, arg.end_date, arg.timespan, arg.multiplier]
   # generate_features_data()
    data()

