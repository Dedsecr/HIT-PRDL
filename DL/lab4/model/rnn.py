import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=64, output_size=32):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
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
            x = input[:, t, :]
            hidden = torch.tanh(
                torch.mm(x, self.W_xh) + torch.mm(hidden, self.W_hh) +
                self.b_h)
            output = torch.mm(hidden, self.W_hy) + self.b_y
            outputs.append(output)
        return outputs, hidden

    def init_hidden(self, batch_size):
        if self.W_xh.device == 'cpu':
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(batch_size, self.hidden_size).cuda()


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=64, output_size=10):
        super(RNNClassifier, self).__init__()
        self.rnn = RNN(input_size, hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, hidden = self.rnn(input)
        output = self.fc(output[-1])
        output = F.log_softmax(output, dim=1)
        return output, hidden


if __name__ == '__main__':
    rnn = RNNClassifier(100).cuda()
    input = torch.randn(32, 10, 100).cuda()
    output, hidden = rnn(input)
    print(output.size())
