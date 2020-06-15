# Breakout Neural Net for Financial Market Prediction
Neural net designed to detect breakouts on financial markets in a certain timeframe above specified standard deviations.

Uses a combination of technical indicators to train for specific price points in a specified timeframe. 

Training data consists of 1min intraday timebars.

Includes script to pull intraday data from Polygon API through Alpaca Brokerage.

Thesis:
This model uses a combination of Volume Weighted Moving Averages, and Simple Moving Averages along with a momentum oscillator (RSI). As a breakout is defined as a bust through a "resistance" or "support" line, these movements are generally associated with volume. Thus, a discepancy between the SMA and VWMA, along with an abnormal momentum reading, may lead a neural net to find high probabilities of a breakout.
