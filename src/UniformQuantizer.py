import torch
import torch.nn as nn
import torch.nn.functional as F

def round_ste(x):
    return (x.round() - x).detach() + x

class UniformQuantizer(nn.Module):
    def __init__(self, n_bits=4, symmetric=True, channel_wise=False, is_weight=False):
        super().__init__()

        self.n_bits = n_bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.is_weight = is_weight # weight or activaition

        # register_buffer를 통해 초기화
        self.register_buffer('max_q', torch.tensor(2 ** (n_bits - 1) - 1))
        self.register_buffer('min_q', torch.tensor(-2 ** (n_bits - 1)))
        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)

    def init_quantization_params(self, x):

        x_clone = x.clone().detach()

        # channel_wise는 언제 사용할까 -> weight를 양자화 할 때, weight는 채널 마다 제각각이기 때문에
        if self.channel_wise:
            max_w = x_clone.abs().view(x_clone.shape[0], -1).max(1)[0]
        else:
            max_w = torch.max(torch.abs(x_clone))

        if self.symmetric:
            self.delta = (max_w / self.max_q).to(x.device)

            if self.channel_wise:
                self.delta = self.delta.view(-1, 1, 1, 1)
                self.zero_point = torch.zeros_like(self.delta)
            else:
                self.zero_point = torch.tensor(0.0).to(x.device)


    def forward(self, x):

        if self.delta is None:
            self.init_quantization_params(x)

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, self.min_q, self.max_q)

        return self.delta * (x_quant - self.zero_point)
