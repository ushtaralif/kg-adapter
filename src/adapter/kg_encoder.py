import torch
import torch.nn as nn


class KGEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim=256,
                 hidden_dim=512, output_dim=512, dropout=0.1, aggregation="mean"):
        super().__init__()
        self.output_dim  = output_dim
        self.aggregation = aggregation

        self.entity_emb   = nn.Embedding(num_entities  + 1, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(num_relations + 1, emb_dim, padding_idx=0)

        self.fc1 = nn.Linear(3 * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        if aggregation == "attention":
            self.attn = nn.Linear(output_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.entity_emb.weight[1:],   std=0.02)
        nn.init.normal_(self.relation_emb.weight[1:], std=0.02)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, subj_ids, pred_ids, obj_ids, mask):
        s = self.entity_emb(subj_ids)
        p = self.relation_emb(pred_ids)
        o = self.entity_emb(obj_ids)

        x = torch.cat([s, p, o], dim=-1)
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)
        x = x * mask.unsqueeze(-1).float()
        return self._pool(x, mask)

    def _pool(self, x, mask):
        if self.aggregation == "mean":
            return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        elif self.aggregation == "max":
            return (x + (1 - mask.unsqueeze(-1).float()) * -1e9).max(dim=1).values
        elif self.aggregation == "attention":
            scores  = self.attn(x).squeeze(-1) + (1 - mask.float()) * -1e9
            return (torch.softmax(scores, dim=-1).unsqueeze(-1) * x).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")