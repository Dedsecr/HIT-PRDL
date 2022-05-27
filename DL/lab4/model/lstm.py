import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.W_xi, self.W_hi, self.b_i = self._get_three(
            input_size, hidden_size)
        self.W_xf, self.W_hf, self.b_f = self._get_three(
            input_size, hidden_size)
        self.W_xo, self.W_ho, self.b_o = self._get_three(
            input_size, hidden_size)
        self.W_xg, self.W_hg, self.b_g = self._get_three(
            input_size, hidden_size)

    def _get_three(self, input_size, output_size):
        W1 = nn.parameter.Parameter(torch.randn(input_size, output_size))
        W2 = nn.parameter.Parameter(torch.randn(output_size, output_size))
        b = nn.parameter.Parameter(torch.randn(output_size))
        return W1, W2, b

    def forward(self, input, state):
        h, c = state
        outputs = []
        for t in range(input.size(0)):
            x = input[:, t, :]
            i = torch.sigmoid(
                torch.mm(x, self.W_xi) + torch.mm(h, self.W_hi) + self.b_i)
            f = torch.sigmoid(
                torch.mm(x, self.W_xf) + torch.mm(h, self.W_hf) + self.b_f)
            o = torch.sigmoid(
                torch.mm(x, self.W_xo) + torch.mm(h, self.W_ho) + self.b_o)
            g = torch.tanh(
                torch.mm(x, self.W_xg) + torch.mm(h, self.W_hg) + self.b_g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)
        return outputs, (h, c)

    def init_hidden(self, batch_size):
        if self.W_xg.device == 'cpu':
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(batch_size, self.hidden_size).cuda()


class LSTMClassifier(nn.Module):
    
        def __init__(self, input_size, hidden_size=64, output_size=10):
            super(LSTMClassifier, self).__init__()
            self.rnn = LSTM(input_size, hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, input):
            output, hidden = self.rnn(input)
            output = self.fc(output[-1])
            output = F.log_softmax(output, dim=1)
            return output, hidden