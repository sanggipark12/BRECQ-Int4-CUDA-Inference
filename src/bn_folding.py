import torch
import torch.nn as nn

def bn_folding(conv, bn):
    # bn params
    mu = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    # conv params
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros_like(mu)

    denominator = torch.sqrt(var + eps)

    w_new = w * (gamma / denominator).view(-1, 1, 1, 1)
    b_new = (b - mu) * (gamma / denominator) + beta

    conv.weight.data.copy_(w_new)

    if conv.bias is not None:
        conv.bias.data.copy_(b_new)
    else:
        conv.bias = nn.Parameter(b_new)

    return nn.Identity()