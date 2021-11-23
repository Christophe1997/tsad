from torch import nn


class RNNModel(nn.Module):

    def __init__(self, in_w, out_w, hidden_dim, num_layers, dropout=0.5, rnn_type="LSTM"):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_window = in_w
        self.out_window = out_w
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(in_w, hidden_dim, num_layers, dropout=dropout)
        else:
            raise ValueError(f"rnn type {rnn_type} not support")

        self.fc = nn.Linear(hidden_dim, out_w)

    def forward(self, x, hidden):
        batch_size = x.shape[1]
        res, hidden = self.rnn(x, hidden)
        res = self.dropout(res)
        res = res.view(-1, self.hidden_dim)
        res = self.fc(res)
        res = res.view(-1, batch_size, self.out_window)
        res = res[-1, :]
        return res, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        init = lambda: weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        if self.rnn_type == "LSTM":
            return init(), init()
        else:
            return init()

    def init_weight(self):
        nn.init.orthogonal_(self.rnn.weight)
        nn.init.kaiming_normal_(self.fc)
