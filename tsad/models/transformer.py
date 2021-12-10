import math

import torch
from torch import nn


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
        self.n_features = n_features

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

        src, _ = self.encoder(src, with_out=True)

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

        return out
