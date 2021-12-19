import torch
from torch import nn
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(RNNEncoder, self).__init__()
        self.rnn1 = getattr(nn, rnn_type)(n_features, hidden_dim, batch_first=True)
        self.rnn2 = getattr(nn, rnn_type)(hidden_dim, emb_dim, batch_first=True)
        self.active1 = nn.ReLU()
        self.active2 = nn.Tanh()

    def forward(self, x, with_out=False):
        # [B, L, N]
        out, _ = self.rnn1(x)
        out = self.active1(out)
        out, hidden = self.rnn2(out)

        if with_out:
            return out, hidden
        else:
            # [1, B, emb_dim]
            return self.active2(hidden[0])


class RNNDecoder(nn.Module):
    def __init__(self, seq_len, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(RNNDecoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.rnn1 = getattr(nn, rnn_type)(emb_dim, emb_dim, batch_first=True)
        self.rnn2 = getattr(nn, rnn_type)(emb_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_features)
        self.active1 = nn.ReLU()

    def forward(self, x):
        x = x.repeat(self.seq_len, 1, 1)
        x = x.transpose(0, 1)
        out, _ = self.rnn1(x)
        out = self.active1(out)
        out, _ = self.rnn2(out)
        out = self.linear(out)

        return out


class RNNAutoEncoder(nn.Module):
    def __init__(self, window_size, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(RNNAutoEncoder, self).__init__()
        self.encoder = RNNEncoder(emb_dim, hidden_dim, rnn_type, n_features=n_features)
        self.decoder = RNNDecoder(window_size, emb_dim, hidden_dim, rnn_type, n_features=n_features)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class LSTMAutoEncoder(nn.Module):
    """Original paper, https://arxiv.org/abs/1607.00148

    References:
        https://github.com/KDD-OpenSource/DeepADoTS/blob/88c38320141a1062301cc9255f3e0fc111f55e80/src/algorithms/lstm_enc_dec_axl.py#L119
    """

    def __init__(self, hidden_dim, num_layers=2, n_features=1, dropout=0.1):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, n_features)

    def forward(self, x, return_latent=False):
        # [B, L, N] -> [1, B, hidden_dim]
        _, enc_hidden = self.encoder(x)
        dec_hidden = enc_hidden
        output = Variable(torch.Tensor(x.size()).zero_())
        for i in reversed(range(x.shape[1])):
            # [B, hidden_dim]
            output[:, i, :] = self.hidden2out(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return output if not return_latent else (output, enc_hidden[1][-1])
