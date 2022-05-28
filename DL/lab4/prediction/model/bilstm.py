import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, device):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def _forward(self, x, h):
        h, cx = h

        gates = self.x2h(x) + self.h2h(h)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)

        return (hy, cy)

    def forward(self, input):
        output, output_rev = [], []
        hidden_forward = self.init_hidden(input.size(0))
        hidden_backward = self.init_hidden(input.size(0))
        hidden_forward = (hidden_forward, hidden_forward)
        hidden_backward = (hidden_backward, hidden_backward)
        for t in range(input.size(1)):
            hidden_forward = self._forward(input[:, t, :], hidden_forward)
            hidden_backward = self._forward(input[:, -t - 1, :],
                                            hidden_backward)
            output.append(hidden_forward[0])
            output_rev.append(hidden_backward[0])
        output = torch.stack(output, dim=1)
        output_rev = torch.stack(output_rev, dim=1)
        output = torch.cat((output, output_rev), dim=2)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(self.device)


class BiLSTMRegressor(nn.Module):

    def __init__(self,
                 input_size=4,
                 target_size=2,
                 hidden_size=64,
                 device='cuda'):
        super(BiLSTMRegressor, self).__init__()
        self.bilstm = BiLSTM(input_size, hidden_size, device=device)
        self.fc = nn.Linear(hidden_size * 2, target_size)

    def forward(self, input):
        output = self.bilstm(input)
        output = output[:, -1, :]
        output = self.fc(output)
        return output