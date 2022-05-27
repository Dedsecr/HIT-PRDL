import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.W_xi, self.W_hi, self.b_i = self._get_three(
            input_size, hidden_size)
        self.W_xf, self.W_hf, self.b_f = self._get_three(
            input_size, hidden_size)
        self.W_xo, self.W_ho, self.b_o = self._get_three(
            input_size, hidden_size)
        self.W_xg, self.W_hg, self.b_g = self._get_three(
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

    def forward(self, input, state):
        h_i, c_i = state
        outputs = []
        for t in range(input.size(0)):
            x_i = input[t]
            i = torch.sigmoid(
                torch.mm(x_i, self.W_xi) + torch.mm(h_i, self.W_hi) + self.b_i)
            f = torch.sigmoid(
                torch.mm(x_i, self.W_xf) + torch.mm(h_i, self.W_hf) + self.b_f)
            o = torch.sigmoid(
                torch.mm(x_i, self.W_xo) + torch.mm(h_i, self.W_ho) + self.b_o)
            g = torch.tanh(
                torch.mm(x_i, self.W_xg) + torch.mm(h_i, self.W_hg) + self.b_g)
            c_i = f * c_i + i * g
            h_i = o * torch.tanh(c_i)
            output = torch.mm(h_i, self.W_hy) + self.b_y
            outputs.append(output)
        return outputs, (h_i, c_i)