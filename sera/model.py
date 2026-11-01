"""Compact assembly of the four archived modules; not the historical experiment runner."""
import torch
from torch import nn
from recbole.model.layers import TransformerEncoder
from .semantic import fused_embeddings
from .encoder import encode_history
from .matching import select_recent, local_scores
from .pooling import aggregate_scores


class SemanticRecentAggregator(nn.Module):
    def __init__(self, metadata_features, metadata_available, config):
        super().__init__()
        d = int(config["hidden_size"])
        features = torch.as_tensor(metadata_features, dtype=torch.float32).clone()
        available = torch.as_tensor(metadata_available, dtype=torch.bool).clone()
        if features.ndim != 2 or features.shape[1] != 384:
            raise ValueError("metadata_features must have shape [N+1, 384]")
        if available.shape != (features.shape[0],):
            raise ValueError("metadata_available must have shape [N+1]")
        available[0] = False
        self.register_buffer("metadata_features", features)
        self.register_buffer("metadata_available", available)
        self.method = "metadata_lse"
        self.recent = int(config["recent"])
        self.alpha = float(config["alpha"])
        self.tau = float(config["tau"])
        self.max_length = int(config["max_length"])
        if self.recent < 1 or self.tau <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError("Require recent>=1, tau>0 and 0<=alpha<=1")
        self.item_embedding = nn.Embedding(features.shape[0], d, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_length, d)
        self.LayerNorm = nn.LayerNorm(d, eps=1e-6)
        self.dropout = nn.Dropout(config["dropout"])
        self.meta_adapter = nn.Linear(384, d, bias=False)
        self.trm_encoder = TransformerEncoder(
            n_layers=2, n_heads=2, hidden_size=d, inner_size=2*d,
            hidden_dropout_prob=config["dropout"], attn_dropout_prob=config["dropout"],
            hidden_act="gelu", layer_norm_eps=1e-6)
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.meta_adapter.weight.zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, seq, item_chunk=2048):
        """Right-padded item IDs [B,L] -> logits [B,N], columns map to IDs 1..N."""
        if seq.ndim != 2 or not 0 < seq.shape[1] <= self.max_length:
            raise ValueError("Require [B,L] with 1<=L<=max_length")
        lengths = (seq != 0).sum(1)
        if torch.any(lengths == 0):
            raise ValueError("Histories must be nonempty")
        expected = torch.arange(seq.shape[1], device=seq.device)[None] < lengths[:, None]
        if not torch.equal(seq != 0, expected):
            raise ValueError("Use contiguous valid IDs followed by right padding")
        if item_chunk < 1:
            raise ValueError("item_chunk must be positive")
        embeddings = fused_embeddings(self)
        h = encode_history(self, seq, embeddings)
        states, valid = select_recent(h, seq, self.recent)
        scores = []
        for start in range(1, embeddings.shape[0], item_chunk):
            z = local_scores(states, embeddings[start:start+item_chunk])
            scores.append(aggregate_scores(z, valid, self.alpha, self.tau))
        return torch.cat(scores, dim=1)
