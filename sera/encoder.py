"""模块2：语义增强历史的因果SASRec编码。"""
import torch

def encode_history(model, seq, embeddings):
    """seq: [B,L]，有效ID在左、右侧补0；输出 H: [B,L,d]。"""
    pos = torch.arange(seq.shape[1], device=seq.device)[None].expand_as(seq)
    validseq = seq != 0
    mask = validseq[:, None, None, :].expand(-1, 1, seq.shape[1], -1).tril()
    h = model.dropout(model.LayerNorm(embeddings[seq] + model.position_embedding(pos)))
    h = model.trm_encoder(h, torch.where(mask, 0., -10000.), output_all_encoded_layers=False)[-1]
    return h
