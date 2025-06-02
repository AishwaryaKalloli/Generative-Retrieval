import torch
import torch.nn as nn

class SemanticIDTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        # note: batch_first=True so inputs are (B, S, E)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                tgt_mask=None):
        B, L_src = src.shape
        _, L_tgt = tgt.shape

        # build position indices - could use a custom postion class 
        src_pos = torch.arange(L_src, device=src.device).unsqueeze(0).expand(B, L_src)
        tgt_pos = torch.arange(L_tgt, device=tgt.device).unsqueeze(0).expand(B, L_tgt)

        src = self.token_embed(src) + self.pos_embed(src_pos)
        tgt = self.token_embed(tgt) + self.pos_embed(tgt_pos)

        # pass masks explicitly
        out = self.transformer(
            src, tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.output_layer(out)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x
