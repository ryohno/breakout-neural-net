import pandas as pd
from pandas_ta import vwma
from pandas_ta import sma
from pandas_ta import rsi
from pandas_ta import bbands
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


# importing training data sets
from torch.autograd import Variable

data_cleaned = pd.read_csv('data_cleaned.csv')
data_cleaned = data_cleaned.assign(slow_vwma_value=(vwma(data_cleaned.close, data_cleaned.volume, length=2, offset=0)),
                                   fast_vwma_value=(vwma(data_cleaned.close, data_cleaned.volume, length=5, offset=0)),
                                   slow_sma_value=(sma(data_cleaned.close, 15)),
                                   fast_sma_value=(sma(data_cleaned.close, 5)),
                                   rsi_data_value=(rsi(data_cleaned.close, 10)),
                                   )
bbands_train = pd.DataFrame(bbands(data_cleaned.close, length=30, std=1.7))
data_cleaned = pd.concat([data_cleaned, bbands_train], axis=1)
# data_cleaned['timestamp'] = pd.to_datetime(data_cleaned['timestamp'], format='%Y-%m-%d %H')
data_cleaned = data_cleaned.set_index('timestamp')
data_cleaned = data_cleaned.dropna()
# print(data_cleaned)
# print(len(data_cleaned))

# importing test data sets
data_cleaned_test = pd.read_csv("data_cleaned_test.csv")
data_cleaned_test = data_cleaned_test.assign(
    slow_vwma_value=(vwma(data_cleaned_test.close, data_cleaned_test.volume, length=15, offset=0)),
    fast_vwma_value=(vwma(data_cleaned_test.close, data_cleaned_test.volume, length=5, offset=0)),
    slow_sma_value=(sma(data_cleaned_test.close, 15)),
    fast_sma_value=(sma(data_cleaned_test.close, 5)),
    rsi_data_value=(rsi(data_cleaned_test.close, 10)),
)
bbands_test = pd.DataFrame(bbands(data_cleaned_test.close, length=30, std=1.7))
data_cleaned_test = pd.concat([data_cleaned_test, bbands_test], axis=1)
# data_cleaned_test['timestamp'] = pd.to_datetime(data_cleaned_test['timestamp'], format='%Y-%m-%d %H')
data_cleaned_test = data_cleaned_test.set_index('timestamp')
data_cleaned_test = data_cleaned_test.dropna()


# print(data_cleaned_test)
# print(len(data_cleaned_test))
def classify(bblower, bbupper, future):
    if future > bbupper:
        return 2
    if future < bblower:
        return 0
    else:
        return 1


# future
future_period_predict = 15
sequence_length = 30
# x and y datasets for TRAIN
future_df_train = data_cleaned['close'].shift(-future_period_predict)
x_train = data_cleaned.reset_index()
del x_train['timestamp']
y_train = pd.DataFrame(columns=["target"])
y_train['target'] = list(map(classify, data_cleaned['BBL_30'], data_cleaned['BBU_30'], future_df_train))
y_train = y_train.to_numpy(dtype=float)
y_train = torch.LongTensor(y_train)

# x and y datasets for TEST
future_df_test = data_cleaned_test['close'].shift(-future_period_predict)
x_test = data_cleaned_test.reset_index()
del x_test['timestamp']
y_test = pd.DataFrame(columns=["target"])
y_test['target'] = list(map(classify, data_cleaned_test['BBL_30'], data_cleaned_test['BBU_30'], future_df_test))
y_test = y_test.to_numpy(dtype=float)
y_test = torch.LongTensor(y_test)

# scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = torch.FloatTensor(x_train)


# creating sequences
def create_sequence(data, y_data, tw):
    sequence = []
    l = len(data)
    for i in range(l - tw):
        train_seq = data[i:i + tw]
        train_label = y_data[i + tw:i + tw + 1]
        sequence.append((train_seq, train_label))
    return sequence


seq = create_sequence(x_train, y_train, 30)


# seq is a 2d torch of [30,13]

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


model = LSTM()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 200
print(model)

for i in range(epochs):
    model.train(mode=True)
    for s, labels in seq:
        s = s.reshape(1, 30, 13)
        optimizer.zero_grad()
        y_pred = model(s)
        loss = loss_function(y_pred, labels[0])
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {loss.item():10.10f}')
