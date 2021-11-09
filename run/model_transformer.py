"""
model_transformer.py: basic Transformer model
Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html

Adapated from Muschick (2020) "Learn2Sign: Sign Language Recognition and Translation
using Human Keypoint Estimation and Transformer Model"
"""

from __future__ import unicode_literals, division
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import torch.nn.functional
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        super(TransformerModel, self).__init__()

        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.decoder_emb = nn.Embedding(ntoken, self.ninp)
        self.pos_decoder = PositionalEncoding(self.ninp, dropout)

        ### FULL TRANSFORMER
        self.transformer = nn.Transformer(d_model=self.ninp, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=self.ninp, dropout=dropout,
                                          activation='relu')

        ### COMBINED TRANSFORMER
        encoder_layer = TransformerEncoderLayer(self.ninp, nhead, nhid, dropout)
        encoder_norm = LayerNorm(self.ninp)
        self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

        # OUTPUT
        self.fc_out = nn.Linear(ninp, ntoken)

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        # key_padding_mask: if provided, specified padding elements in the key will be ignored by the attention.
        # This is an binary mask. When the value is True, the corresponding value on the attention layer will be filled with -inf.
        # 0 / False = real value
        # 1 / True = padding token
        return (inp == 0).transpose(0, 1)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder_emb.bias.data.zero_()
        self.decoder_emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):

        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            self.tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        padding_tensor = src.mean(2)
        src_pad_mask = self.make_len_mask(padding_tensor)
        src_pad_mask = src_pad_mask.permute(1, 0)

        tgt_pad_mask = self.make_len_mask(tgt)
        tgt_pad_mask = tgt_pad_mask.permute(1, 0)

        src = self.pos_encoder(src)

        tgt = self.decoder_emb(tgt)
        tgt = self.pos_decoder(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        output = self.encoder(src, mask=self.src_mask)
        output = self.fc_out(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
