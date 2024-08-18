import math
from typing import Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_attn_pad_mask(seq_k: Tensor, pad_token_id: int = 0, device: Optional[torch.device] = None) -> Tensor:
    pad_attn_mask = (seq_k == pad_token_id).unsqueeze_(1).to(device)
    return pad_attn_mask


def get_attn_subsequent_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool, device=device), 1)
    return mask


def cal_positional_encoding(max_len: int, embed_dim: int) -> Tensor:
    idx = torch.arange(0, embed_dim, 2, dtype=torch.float)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1)
    div_term = torch.exp(-idx / embed_dim * math.log(10000))
    pos = pos * div_term

    pe = torch.zeros(max_len, embed_dim)
    pe[:, 0::2] = torch.sin(pos)
    pe[:, 1::2] = torch.cos(pos)
    return pe


def cal_attntion(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None, dropout: nn.Dropout = None) -> Tuple[Tensor, Tensor]:
    embed_dim = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_dim)
    if mask is not None:
        mask = mask.unsqueeze(1)  # 为了能在 nhead 维度上进行广播
        scores.masked_fill_(mask, -1e9)
    attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention = dropout(attention)
    x = torch.matmul(attention, V)
    return x, attention


def create_repeat_layers(layer_type: Type, num_layers: int, *args, **kwargs) -> nn.ModuleList:
    repeated_layers = nn.ModuleList([layer_type(*args, **kwargs) for _ in range(num_layers)])
    return repeated_layers


def expand_mask(mask: Tensor, nhead: int, length: int = None) -> Tensor:
    bs, ntoken = mask.size(0), mask.size(-1)
    if length is None:
        length = ntoken
    mask = mask.unsqueeze(1).expand(bs, nhead, length, ntoken)
    return mask.reshape(bs * nhead, length, ntoken)


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.dropout = nn.Dropout(dropout)
        self.lut = nn.Embedding(vocab_size, embed_dim)
        pe = cal_positional_encoding(max_len, embed_dim)
        pe.unsqueeze_(0)
        self.register_buffer("pe", pe)
        self.pe: Tensor

    def forward(self, x: torch.Tensor) -> Tensor:
        # TODO 为什么要乘 sqrt(self.embed_dim)
        x = self.lut(x) * math.sqrt(self.embed_dim)
        x = self.dropout(x + self.pe[:, : x.size(1)])
        return x


class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()

        assert embed_dim % nhead == 0
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        bs = len(query)
        (Q, K, V) = (linear(x).reshape(bs, -1, self.nhead, self.head_dim).transpose(1, 2) for linear, x in zip((self.w_q, self.w_k, self.w_v), (query, key, value)))
        output, attention = cal_attntion(Q, K, V, mask, self.dropout)
        output = output.transpose(1, 2).reshape(bs, -1, self.embed_dim)

        # output: [batch_size x len_q x embed_dim]
        output = self.w_o(output)
        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, ff_dim: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttn(embed_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm1(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        output = self.dropout(self.self_attn(x, x, x, attn_mask)[0])
        return output

    def _ff_block(self, x: Tensor) -> Tensor:
        output = self.dropout(self.ff(x))
        return output


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, ff_dim: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttn(embed_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttn(embed_dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_mask: Tensor) -> Tensor:
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask))
        x = self.norm2(x + self._ca_block(x, memory, memory_mask))
        x = self.norm3(x + self.ff(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.dropout(self.self_attn(x, x, x, attn_mask)[0])
        return x

    def _ca_block(self, x: Tensor, mem: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.dropout(self.cross_attn(x, mem, mem, attn_mask)[0])
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.dropout(self.ff(x))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, nhead: int, ff_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.layers = create_repeat_layers(EncoderLayer, num_layers, embed_dim, nhead, ff_dim, dropout)
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class Decoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, nhead: int, ff_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.layers = create_repeat_layers(DecoderLayer, num_layers, embed_dim, nhead, ff_dim, dropout)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_mask: Tensor) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.encoder = Encoder(num_encoder_layers, embed_dim, nhead, ff_dim, dropout)
        self.decoder = Decoder(num_decoder_layers, embed_dim, nhead, ff_dim, dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        return output

    def encode(self, src: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.encoder.forward(src, attn_mask)
        return x

    def decode(self, tgt: Tensor, mem: Tensor, tgt_mask: Tensor, mem_mask: Tensor) -> Tensor:
        x = self.decoder.forward(tgt, mem, tgt_mask, mem_mask)
        return x

    def generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None) -> Tensor:
        mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)
        return mask


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        ff_dim: int,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.tgt_mask: Tensor = None
        self.transformer = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, ff_dim, dropout)
        # self.transformer = nn.Transformer(
        #     embed_dim,
        #     nhead,
        #     num_encoder_layers,
        #     num_decoder_layers,
        #     ff_dim,
        #     dropout,
        #     batch_first=True,
        # )
        self.pos_encoder = PositionalEncoding(vocab_size, embed_dim, dropout)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        # memory_mask = expand_mask(src_mask, self.nhead, tgt_mask.size(-1))
        memory_mask = None
        # src_mask = expand_mask(src_mask, self.nhead)
        # tgt_mask = expand_mask(tgt_mask, self.nhead)
        output = self.transformer.forward(src, tgt, src_mask, tgt_mask)
        output = self.linear(output)
        return output

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder.forward(self.pos_encoder(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, mem_mask: Tensor) -> Tensor:
        return self.transformer.decoder.forward(self.pos_encoder(tgt), memory, tgt_mask, mem_mask)

    def generate(self, x: Tensor) -> Tensor:
        return self.linear(x)
