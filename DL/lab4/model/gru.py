import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size=64, output_size=32):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.W_xr, self.W_hr, self.b_r = self._get_three(
            input_size, hidden_size)
        self.W_xz, self.W_hz, self.b_z = self._get_three(
            input_size, hidden_size)
        self.W_xh, self.W_hh, self.b_h = self._get_three(
            input_size, hidden_size)
        self.W_hy, self.b_y = self._get_two(hidden_size, output_size)

    def _get_three(self, input_size, output_size):
        W1 = nn.parameter.Parameter(torch.randn(input_size, output_size))
        W2 = nn.parameter.Parameter(torch.randn(output_size, output_size))
        b = nn.parameter.Parameter(torch.randn(output_size))
        return W1, W2, b

    def _get_two(self, input_size, output_size):
        W = nn.parameter.Parameter(torch.randn(input_size, output_size))
        b = nn.parameter.Parameter(torch.randn(output_size))
        return W, b

    def forward(self, input):
        outputs = []
        hidden = self.init_hidden(input.size(0))
        for t in range(input.size(1)):
            r = F.sigmoid(
                torch.mm(input[t], self.W_xr) + torch.mm(hidden, self.W_hr) +
                self.b_r)
            z = F.sigmoid(
                torch.mm(input[t], self.W_xz) + torch.mm(hidden, self.W_hz) +
                self.b_z)
            h_hat = F.tanh(
                torch.mm(input[t], self.W_xh) +
                torch.mm(hidden * r, self.W_hh) + self.b_h)
            hidden = (1 - z) * hidden + z * h_hat
            output = torch.mm(hidden, self.W_hy) + self.b_y
            outputs.append(output)
        return outputs, hidden

    def init_hidden(self, batch_size):
        if self.W_xh.device == 'cpu':
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(batch_size, self.hidden_size).cuda()


class GRUClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=64, output_size=10):
        super(GRUClassifier, self).__init__()
        self.gru = GRU(input_size, hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        output = self.fc(hidden)
        output = F.log_softmax(output, dim=1)
        return output, hidden