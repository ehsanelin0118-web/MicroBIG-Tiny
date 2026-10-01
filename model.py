import torch
import torch.nn as nn
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizer.json')
VOCAB_SIZE = tok.get_vocab_size()

EMB_DIM = 128
FFN_DIM = 512
NUM_LAYERS = 4
SEQ_LEN = 128

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, ffn, s_len):
        super().__init__()
        self.register_buffer(
        "causal_mask",
        torch.triu(
            torch.ones(s_len, s_len),
            diagonal=1
        ).bool()
        )

        self.attn = nn.MultiheadAttention(emb_dim,num_heads=16, batch_first=True, dropout=0.01)
        self.norm = nn.LayerNorm(emb_dim)

        self.ffn1 = nn.Linear(emb_dim,ffn)
        self.ffn2 = nn.Linear(ffn,emb_dim)
    def forward(self, x):
        T = x.size(1)
        self.a, weights = self.attn(x, x, x, attn_mask=self.causal_mask[:T, :T])
        self.b = self.norm(self.a)

        self.c = self.ffn1(self.b)
        self.d = self.ffn2(self.c)
        d = self.d + self.b

        return d

class Model(nn.Module):
    def __init__(self, emb_dim, v_size, n_layers, s_len, ffn):
        super().__init__()

        self.s_len = s_len

        self.embedding = nn.Embedding(v_size, emb_dim)
        self.position = nn.Embedding(s_len, emb_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim,ffn,s_len)
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(emb_dim,v_size)

        self.lm_head.weight = self.embedding.weight
        self.dr = nn.Dropout(0.08)

    def forward(self, x):
        a = self.dr(self.embedding(x))
        T = x.size(1)
        positions = torch.arange(T).cuda()
        
        p = self.position(positions)
        a = a + p

        for l in self.layers:
            a = l(a)

        return self.lm_head(a)




model = Model(
    EMB_DIM,
    VOCAB_SIZE,
    NUM_LAYERS,
    SEQ_LEN,
    FFN_DIM
).cuda()

params = sum(p.numel() for p in model.parameters())

print('Model Created')
print(f"Parameters: {params}")