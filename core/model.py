import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        T = h.size(1)
        return h + self.pe[:T].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**0.5

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * Dh)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        Q = self.split_heads(self.Wq(h))
        K = self.split_heads(self.Wk(h))
        V = self.split_heads(self.Wv(h))

        scores = (Q @ K.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e9)
            q_pad = key_padding_mask[:, None, :, None]
            scores = scores.masked_fill(q_pad, 0.0)

        attn = softmax_stable(scores, dim=-1)

        if key_padding_mask is not None:
            q_pad = key_padding_mask[:, None, :, None]
            attn = attn.masked_fill(q_pad, 0.0)

        attn = self.drop(attn)
        out = attn @ V
        out = self.combine_heads(out)
        out = self.Wo(out)
        out = self.drop(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), key_padding_mask)

        x = self.ln2(h)
        x = self.ff2(self.drop(F.gelu(self.ff1(x))))
        x = self.drop(x)
        h = h + x
        return h


class TransformerEncoderTagger(nn.Module):
    def __init__(
        self,
        vocab_in: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        pad_x: int,
        num_classes: int,
        max_len: int,
    ):
        super().__init__()
        self.pad_x = pad_x
        self.emb = nn.Embedding(vocab_in, d_model, padding_idx=pad_x)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len + 4)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key_padding_mask = x == self.pad_x
        h = self.emb(x)
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        return self.out(h)