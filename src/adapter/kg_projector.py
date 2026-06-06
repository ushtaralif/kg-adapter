import torch
import torch.nn as nn


class KGProjector(nn.Module):
    def __init__(self, input_dim, d_model, num_kg_tokens=16,
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.num_kg_tokens = num_kg_tokens
        self.d_model       = d_model

        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, num_kg_tokens * d_model)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, kg_emb):
        B   = kg_emb.size(0)
        out = self.drop(self.act(self.fc1(kg_emb)))
        return self.fc2(out).view(B, self.num_kg_tokens, self.d_model)