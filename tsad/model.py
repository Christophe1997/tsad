import math

import torch
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
        hidden = hidden.view(-1, self.hidden_dim)
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
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        nn.init.kaiming_normal_(self.fc.weight)


class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(Encoder, self).__init__()
        self.rnn1 = getattr(nn, rnn_type)(n_features, hidden_dim, batch_first=True)
        self.rnn2 = getattr(nn, rnn_type)(hidden_dim, emb_dim, batch_first=True)
        self.active1 = nn.ReLU()
        self.active2 = nn.Tanh()

    def forward(self, x, with_out=False):
        # [B, L] -> [B, L, N]
        x = x.unsqueeze(-1)
        out, _ = self.rnn1(x)
        out = self.active1(out)
        out, hidden = self.rnn2(out)

        if with_out:
            return out, hidden
        else:
            # [1, B, emb_dim]
            return self.active2(hidden[0])


class Decoder(nn.Module):
    def __init__(self, seq_len, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
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
        out = out.squeeze(-1)

        return out


class AutoEncoder(nn.Module):
    def __init__(self, window_size, emb_dim, hidden_dim, rnn_type="GRU", n_features=1):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(emb_dim, hidden_dim, rnn_type, n_features=n_features)
        self.decoder = Decoder(window_size, emb_dim, hidden_dim, rnn_type, n_features=n_features)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        x = x + self.pe[:x.size(0)]
        res = self.dropout(x)

        return res if not self.batch_first else res.transpose(0, 1)


class LinearDTransformer(nn.Module):

    def __init__(self, d_model, history_w, predict_w, encoder=None,
                 n_features=1,
                 nhead=8,
                 nlayers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 overlap=True,
                 batch_first=True):
        super(LinearDTransformer, self).__init__()
        self.batch_first = batch_first
        self.seq_len = history_w
        self.overlap = overlap

        self.pos_encoder = PositionalEncoding(d_model, dropout, batch_first=batch_first)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        if encoder is not None:
            self.encoder = encoder
            self.encoder.requires_grad_(False)
        else:
            self.encoder = nn.Linear(n_features, d_model)

        self.decoder = nn.Linear(d_model, n_features)
        if not overlap:
            self.fc = nn.Linear(history_w, predict_w)
        else:
            self.fc = None

        self.init_weight()

    def init_weight(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.fc is not None:
            self.fc.bias.data.zero_()
            self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        # [B, L] -> [B, emb_dim]
        src = self.encoder(src)
        if len(src.shape) < 3:
            src = src.repeat(self.seq_len, 1, 1)
            src = src.transpose(0, 1)

        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, src_mask)
        out = self.decoder(out)

        if not self.overlap:
            if self.batch_first:
                # [B, L, N] -> [B, N, L]
                out = out.permute(0, 2, 1)
            else:
                # [L, B, N] -> [B, N, L]
                out = out.permute(1, 2, 0)

            out = self.fc(out)
            # [B, N, L] -> [B, L, N]
            out = out.permute(0, 2, 1)

        return out.squeeze(-1)
