"""Synthetic one-step wiring example; this does not reproduce benchmark metrics."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from sera import SemanticRecentAggregator


def main():
    torch.manual_seed(42)
    config = json.loads((Path(__file__).parent / "configs/video_games.json").read_text())
    features = F.normalize(torch.randn(21, 384), dim=-1)
    available = torch.ones(21, dtype=torch.bool)
    model = SemanticRecentAggregator(features, available, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    seq = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    target_ids = torch.tensor([6, 7])
    optimizer.zero_grad()
    scores = model(seq)
    loss = F.cross_entropy(scores, target_ids - 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
    optimizer.step()
    print("Synthetic one-step example:", tuple(scores.shape), float(loss.detach()))


if __name__ == "__main__":
    main()
