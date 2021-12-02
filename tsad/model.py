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
            self.rnn = getattr(nn, rnn_type)(1, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"rnn type {rnn_type} not support")

        self.fc = nn.Linear(hidden_dim, out_w)

    def forward(self, x):
        x = x.unsqueeze(-1)

        out, hidden = self.rnn(x)
        hidden = hidden[0]
        hidden = hidden[-1, :].view(-1, self.hidden_dim)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        init = lambda: weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        if self.rnn_type == "LSTM":
            return init(), init()
        else:
            return init()

    def init_weight(self):
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        nn.init.kaiming_normal_(self.fc.weight)


class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(Encoder, self).__init__()
        self.rnn1 = getattr(nn, rnn_type)(n_features, hidden_dim, batch_first=True)
        self.rnn2 = getattr(nn, rnn_type)(hidden_dim, emb_dim, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn1(x)
        out, hidden = self.rnn2(out)

        return hidden[0]


class Decoder(nn.Module):
    def __init__(self, seq_len, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.rnn1 = getattr(nn, rnn_type)(emb_dim, emb_dim)
        self.rnn2 = getattr(nn, rnn_type)(emb_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1, 1)
        x = x.transpose(0, 1)
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = self.linear(out)
        out = out.squeeze()
        return out


class AutoEncoder(nn.Module):
    def __init__(self, out_w, hidden_dim, rnn_type="GRU"):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(out_w, hidden_dim, rnn_type)
        self.decoder = Decoder(out_w, hidden_dim, rnn_type)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(*out)
        return out
