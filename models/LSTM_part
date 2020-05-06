import torch.nn as nn
import torch


class seg_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=3, bi_directional=True, predict_class=100):
        super(seg_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=bi_directional)
        self.linear = nn.Linear(hidden_size, predict_class)
        self.softmax = nn.Softmax(2)

    def forward(self, x, hidden):
        output, hidden_out = self.lstm(x, hidden)
        output = self.linear(output)
        return output, hidden_out
