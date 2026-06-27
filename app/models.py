import torch
import torch.nn as nn

class BaseGRU(nn.Module):
    def __init__(self, num_items, num_events, embedding_dim=64, event_embedding_dim=8,
                 hidden_size=224, num_layers=2, dropout=0.3):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.event_emb = nn.Embedding(num_events, event_embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim + event_embedding_dim, hidden_size,
                          num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + 3, num_items)

    def forward(self, items, events, hour, day, time_gap):
        item_emb = self.item_emb(items)
        event_emb = self.event_emb(events)
        x = torch.cat([item_emb, event_emb], dim=-1)
        x = self.dropout(x)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        context = torch.cat([hour, day, time_gap], dim=1)
        combined = torch.cat([last_hidden, context], dim=1)
        logits = self.fc(combined)
        return logits