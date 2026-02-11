import torch.nn as nn
import torch.nn.functional as F

class QuantModule(nn.Module):
    def __init__(self, org_module: nn.Conv2d, weight_quantizer):
        super().__init__()
        self.org_module = org_module
        self.weight_quantizer = weight_quantizer # 아까 만든 AdaRoundQuantizer
        self.use_quantization = True # 스위치
        self.conv_type = dict(bias = self.org_module.bias,
                              stride = self.org_module.stride, padding = self.org_module.padding,
                                dilation=self.org_module.dilation, groups=self.org_module.groups)

    def forward(self, x):
        if self.use_quantization:
            # 가중치 가져오기
            w = self.org_module.weight

            # 가중치 양자화 (여기서 AdaRound가 작동)
            w_q = self.weight_quantizer(w)

            # 양자화된 가중치로 Conv 연산
            out = F.conv2d(x, w_q, **self.conv_type)
            return out
        else:
            return self.org_module(x)