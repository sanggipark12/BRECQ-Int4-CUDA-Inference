import torch
import torch.nn as nn
import UniformQuantizer

class AdaRoundQuantizer(nn.Module):
    def __init__(self, uq :UniformQuantizer, x):
        super().__init__()

        self.uq = uq
        self.alpha = None # AdaRound parameter (v)
        self.soft_targets = True # adaround 여부

        # 고정 상수들을 buffer로 등록
        self.register_buffer('gamma', torch.tensor(-0.1))
        self.register_buffer('zeta', torch.tensor(1.1))

        self.init_alpha(x.clone())

    def init_alpha(self, x):
        w_floor = torch.floor(x / self.uq.delta)

        rest = (x / self.uq.delta) - w_floor
        rest = torch.clamp(rest, 0.01, 0.99)

        sig_inv = (rest - self.gamma) / (self.zeta - self.gamma)
        sig_inv = torch.clamp(sig_inv, 0.01, 0.99)

        alpha = torch.log(sig_inv / (1 - sig_inv))
        self.alpha = nn.Parameter(alpha)

    def rectified_sigmoid(self):
        # Equation 23
        x = torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

        return x

    def forward(self, x):
        w_floor = torch.floor(x / self.uq.delta)

        # 모드에 따라 0~1 사이의 값(soft) 또는 0/1 값(hard)을 결정
        if self.soft_targets:
            rounding_val = self.rectified_sigmoid()
        else:
            rounding_val = (self.alpha >= 0.5).float()

        w_int = w_floor + rounding_val
        w_quant = self.uq.delta * torch.clamp(w_int - self.uq.zero_point, self.uq.min_q, self.uq.max_q)

        return w_quant