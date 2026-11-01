"""模块1：历史输入与候选侧共用的商品融合表示。"""
def fused_embeddings(model):
    """输出 [N+1, d]；第0行是填充；model沿用原始PilotModel初始化。"""
    e = model.item_embedding.weight
    if model.method == 'metadata_lse':
        e = e + model.meta_adapter(model.metadata_features) * model.metadata_available[:, None]
    return e
