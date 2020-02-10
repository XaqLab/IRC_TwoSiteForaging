"""
-- RNN model class --
Architecture:
Input layer --> Hidden recurrent layer (hidden_size1)
                       --> Linear with RELU (hidden_size2)
                                  --> Linear with softmax (output_size_act = na)-- > action
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size_act, num_layers, param_len, softmax_temp = 1):
        super(RNN, self).__init__()

        self.input_size = input_size  + 1 * param_len
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.output_size_act = output_size_act

        self.rnn = nn.RNN(self.input_size, self.hidden_size1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size1 + 0 * param_len, self.hidden_size2, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size2, self.output_size_act, bias=True)
        self.softmax = nn.Softmax(dim=2)

        self.softmax_temp = softmax_temp

    def forward(self, x1, para, h_0=None):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        if h_0 is None:
            h_0 = Variable(torch.zeros(self.num_layers, x1.size(0), self.hidden_size1))

        combined = torch.cat((x1, para.unsqueeze(1).repeat(1, x1.size()[1], 1)), 2)
        out_neu, _ = self.rnn(combined, h_0)
        out = self.fc1(out_neu)

        #out_neu, _ = self.rnn(x1, h_0)
        #combined = torch.cat((out_neu, x2.unsqueeze(1).repeat(1, x1.size()[1], 1)), 2)
        #out = self.fc1(combined)

        out = self.activation(out)
        out = self.fc2(out)
        out = self.softmax(out / self.softmax_temp)
        return out, out_neu