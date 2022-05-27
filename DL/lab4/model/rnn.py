import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def _forward(self, x, h):
        return torch.tanh(self.x2h(x) + self.h2h(h))

    # def forward(self, input):
    #     output = None
    #     hidden = self.init_hidden(input.size(0))
    #     input = input.permute(1, 0, 2)
    #     for x in input:
    #         hidden = self._forward(x, hidden)
    #         if output is None:
    #             output = hidden.unsqueeze(0)
    #         else:
    #             output = torch.cat([output, hidden.unsqueeze(0)], dim=0)
    #     output = output.permute(1, 0, 2)
    #     return output, hidden

    def forward(self, input):
        outs = []
        hidden = self.init_hidden(input.size(0))
        for t in range(input.size(1)):
            hidden = self._forward(input[:, t, :], hidden)
            outs.append(hidden)
        return torch.stack(outs, dim=1), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(self.device)


# class RNN_(torch.nn.Module):

#     def __init__(self, input_size, hidden_size):
#         super(RNN_, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnncell = nn.RNNCell(input_size, hidden_size)

#     def forward(self, input):
#         output = None
#         hidden = self.init_hidden(input.size(0))
#         input = input.permute(1, 0, 2)
#         for x in input:
#             # print(x.shape)
#             hidden = self.rnncell(x, hidden)
#             if output is None:
#                 output = hidden.unsqueeze(0)
#             else:
#                 output = torch.cat([output, hidden.unsqueeze(0)], dim=0)
#         output = output.permute(1, 0, 2)
#         return output, hidden

#     def init_hidden(self, batch_size):
#         # if self.rnncell.device == 'cpu':
#         #     return torch.zeros(batch_size, self.hidden_size)
#         # else:
#         return torch.zeros(batch_size, self.hidden_size).cuda()

# class MyRNN(torch.nn.Module):

#     def __init__(self, input_size, hidden_size):
#         super(MyRNN, self).__init__()
#         self.hidden_size = hidden_size
#         # self.rnncell = MyRNNcell(256, 10)
#         self.rnncell = nn.RNNCell(input_size, hidden_size)

#     def forward(self, x):
#         x = x.transpose(1, 0)
#         h = self.init_hidden(x.size(1))
#         # print(x.size())
#         output = None
#         for data in x:
#             h = self.rnncell(data, h)
#             if output is None:
#                 output = h.unsqueeze(0)
#             else:
#                 output = torch.cat([output, h.unsqueeze(0)], dim=0)
#         output = output.transpose(1, 0)
#         return output, h

#     def init_hidden(self, batch_size):
#         return torch.zeros(batch_size, self.hidden_size).cuda()


class RNNClassifier(nn.Module):

    def __init__(self,
                 word_num,
                 embedding_size,
                 hidden_size=64,
                 max_length=64,
                 num_classes=10,
                 device='cuda'):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(word_num, embedding_size)
        self.rnn = RNN(embedding_size, hidden_size, device=device)
        self.output_size = hidden_size * max_length
        self.fc = nn.Linear(self.output_size, num_classes)

    def forward(self, input):
        input = self.embedding(input)
        output, hidden = self.rnn(input)
        output = self.fc(output.contiguous().view(-1, self.output_size))
        output = F.log_softmax(output, dim=1)
        return output, hidden
