# Breakout NeuralNet

<div align="center">
<img src=https://raw.githubusercontent.com/ryohno/RyohTradingFramework/master/ryoh%20logo.PNG>
</div>


Neural net designed to detect breakouts on financial markets in a certain timeframe above specified standard deviations. Uses a combination of technical indicators (using Pandas-TA library) to train for specific price points in a specified timeframe.

Thesis: This model uses a combination of Volume Weighted Moving Averages, and Simple Moving Averages along with a momentum oscillator (RSI). As a breakout is defined as a bust through a "resistance" or "support" line, these movements are generally associated with volume. Thus, a discepancy between the SMA and VWMA, along with an abnormal momentum reading, may lead a neural net to find high probabilities of a breakout.


## Datasets

Use the included script:

```python
getdata.py
```
This script allows user to pull intraday data from Polygon API through Alpaca Brokerage. While the API limits requests for data at 500, this script loops through the range of dates given for each day as a workaround to pull all data in one command. For example, a command for intraday SPY data, on one minute intervals, through the dates of June 1 to July 1 would be written as:
```python
python getdata.py SPY 2020-06-01 2020-07-01 minute 1
```
Two datasets, data_cleaned.csv and data_cleaned_test.csv are imported as Pandas Dataframes for training and test data. Using pandas-ta library, volume-weighted moving averages and RSI are added to each Pandas dataframe. Bollinger Bands are also created in a different Pandas Dataframe for later use. 


## Classify Function

```python
def classify(bblower, bbupper, future):
    if future > bbupper:
        return 2
    if future < bblower:
        return 0
    else:
        return 1
```
```python
y_test['target'] = list(map(classify, data_cleaned_test['BBL_30'], data_cleaned_test['BBU_30'], future_df_test))
```
```python
y_train['target'] = list(map(classify, data_cleaned['BBL_30'], data_cleaned['BBU_30'], future_df_train))
```

Targets are created using the classify function, taking the lower Bollinger Band, upper Bollinger Band, and future price value as inputs. If future price is above or lower the respective Bollinger Band, a value of 0 (for below), 2 (for above) is given. The outputs are mapped into y_test and y_train DataFrames.


## Scale Data
```python
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = torch.FloatTensor(x_train)
```
Data is scaled for input.

## Create Sequence
```python
def create_sequence(data, y_data, tw):
    sequence = []
    l = len(data)
    for i in range(l - tw):
        train_seq = data[i:i + tw]
        train_label = y_data[i + tw:i + tw + 1]
        sequence.append((train_seq, train_label))
    return sequence
```
## Model Implementation
```python
class LSTM(nn.Module):
    def __init__(self, input_size=13, hidden_layer_size=100, output_size=3, batch_size=1, time_step=30):
        super().__init__()
        self.time_step = time_step
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.lstm_hidden = nn.LSTM(hidden_layer_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.output_linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(self.batch_size, self.time_step, self.hidden_layer_size),
                            torch.zeros(self.batch_size, self.time_step, self.hidden_layer_size))

    def forward(self, input_seq):
        # print((input_seq.reshape(len(input_seq)*self.input_size)))
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        lstm_hidden_out, self.hidden_cell = self.lstm_hidden(lstm_out)
        linear1 = self.linear(lstm_hidden_out[:, -1, :])
        relu1 = self.relu(linear1)
        linear2 = self.linear(relu1)
        relu2 = self.relu(linear2)
        prediction = self.output_linear(relu2)

        return prediction

```

#



