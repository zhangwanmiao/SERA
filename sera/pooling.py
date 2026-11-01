"""模块4：有效状态评分的Mean与归一化温度化LSE。"""
import torch

def aggregate_scores(z, valid, alpha=0.5, tau=1.0):
    """z: [B,K,C]，valid: [B,K]；返回 [B,C]。tau>0且每条历史非空。"""
    count = valid.sum(1).float().clamp_min(1)
    mean = (z * valid[..., None]).sum(1) / count[:, None]
    lse = tau * (torch.logsumexp(z.masked_fill(~valid[..., None], -torch.inf) / tau, 1)
                 - count.log()[:, None])
    return (1 - alpha) * mean + alpha * lse
