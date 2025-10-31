# model/GIMLET/contrastive.py
import torch
import torch.nn.functional as F


def compute_contrastive_loss(anchor, positive, temperature=0.07):
    """
    anchor: (B, D)
    positive: (B, D)
    returns: scalar loss
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    logits = torch.matmul(anchor, positive.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def multimodal_contrastive_loss(text_repr=None, graph_repr=None, fingerprint_repr=None, temperature=0.07):
    """
    Support multiple contrastive objectives. Only computes those with non-None inputs.
    Returns: total_loss, dict with individual components
    """
    print("fp_repr is None:", fingerprint_repr is None)

    loss_dict = {}
    total_loss = 0.0

    if text_repr is not None and graph_repr is not None:
        loss_tg = compute_contrastive_loss(text_repr, graph_repr, temperature)
        loss_dict['loss_text_graph'] = loss_tg
        total_loss += loss_tg

    if fingerprint_repr is not None and graph_repr is not None:
        loss_fg = compute_contrastive_loss(fingerprint_repr, graph_repr, temperature)
        loss_dict['loss_fingerprint_graph'] = loss_fg
        total_loss += loss_fg

    if fingerprint_repr is not None and text_repr is not None:
        loss_ft = compute_contrastive_loss(fingerprint_repr, text_repr, temperature)
        loss_dict['loss_fingerprint_text'] = loss_ft
        total_loss += loss_ft

    return total_loss, loss_dict
