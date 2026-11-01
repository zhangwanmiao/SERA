"""模块3：选择近期有效状态，再逐候选计算内积。"""
import torch

def select_recent(h, seq, recent=5):
    """输出 [B,K,d]、[B,K]；顺序为最近状态在前。历史至少含1个有效ID。"""
    length = (seq != 0).sum(1)
    ix = length[:, None] - 1 - torch.arange(recent, device=seq.device)[None]
    valid = ix >= 0
    states = h.gather(1, ix.clamp_min(0)[..., None].expand(-1, -1, h.shape[-1]))
    return states, valid

def local_scores(states, candidate_embeddings):
    """[B,K,d] × [C,d] -> [B,K,C]；无效位置交由模块4屏蔽。"""
    return torch.einsum('bkd,nd->bkn', states, candidate_embeddings)
