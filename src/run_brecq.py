import torch
import torch.nn as nn
import QuantModule
import block_reconstruction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EfficientDataSaver:
    def __init__(self, total_samples, in_c, in_h, in_w, out_c, out_h, out_w):
        self.inputs = torch.zeros((total_samples, in_c, in_h, in_w), dtype=torch.float32)
        self.outputs = torch.zeros((total_samples, out_c, out_h, out_w), dtype=torch.float32)
        self.grads = torch.zeros((total_samples, out_c, out_h, out_w), dtype=torch.float32)
        self.idx = 0
        self.backward_idx = 0

    def hook_fn(self, module, input, output):
        batch_size = input[0].shape[0]
        curr_idx = self.idx

        if curr_idx + batch_size <= self.inputs.shape[0]:

            self.inputs[curr_idx : curr_idx + batch_size] = input[0].detach().cpu()
            self.outputs[curr_idx : curr_idx + batch_size] = output.detach().cpu()

            self.idx += batch_size

    def hook_backward(self, module, grad_input, grad_output):
        if grad_output[0] is None:
            return None

        batch_size = grad_output[0].shape[0]
        curr_idx = self.backward_idx

        if curr_idx + batch_size <= self.grads.shape[0]:

            self.grads[curr_idx : curr_idx + batch_size] = grad_output[0].detach().cpu()
            self.backward_idx += batch_size

        return None

def get_quantizers_from_block(block):
    """블록 내부의 모든 QuantModule에서 weight_quantizer를 추출"""
    quantizers = []
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            quantizers.append(module.weight_quantizer)
    return quantizers


def set_quant_state(module, use_quant=True):
    """
    블록 내의 모든 QuantModule의 양자화 스위치를 켜거나 끕니다.
    """
    for m in module.modules():
        if isinstance(m, QuantModule):
            m.use_quantization = use_quant


def run_brecq(model, dataloader, target_block, num_samples=128):
    model.eval().to(device)

    sample_img, _ = next(iter(dataloader))
    sample_img = sample_img[:1].to(device) # 1개만 사용

    block_shape = {}
    def get_shape_hook(module, input, output):
        # input[0]은 입력 텐서, output은 출력 텐서
        block_shape['in'] = input[0].shape[1:]  # (C, H, W)
        block_shape['out'] = output.shape[1:]    # (C, H, W)

    handle = target_block.register_forward_hook(get_shape_hook)
    with torch.no_grad():
        model(sample_img)
    handle.remove()

    in_c, in_h, in_w = block_shape['in']
    out_c, out_h, out_w = block_shape['out']

    # 입력과 출력의 크기가 다를 수 있으므로(stride 등) 각각 할당합니다.
    saver = EfficientDataSaver(num_samples, in_c, in_h, in_w, out_c, out_h, out_w)

    set_quant_state(target_block, use_quant=False)

    # Hook 등록
    # forward: 입력(input)과 정답(output) 수집
    h1 = target_block.register_forward_hook(saver.hook_fn)
    # backward: 중요도(gradient) 수집
    h2 = target_block.register_full_backward_hook(saver.hook_backward)

    criterion = nn.CrossEntropyLoss()

    print("Step 1: Collecting Data & Gradients...")
    current_samples = 0
    for imgs, labels in dataloader:
        if current_samples >= num_samples: break
        imgs, labels = imgs.to(device), labels.to(device)
        imgs.requires_grad_(True)

        model.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()

        current_samples += imgs.shape[0]


    # Hook 제거
    h1.remove()
    h2.remove()

    set_quant_state(target_block, use_quant=True)

    # Quantizer 추출
    quantizers = get_quantizers_from_block(target_block)

    print("Step 2: Optimizing Block...")
    block_reconstruction(
        target_block,
        saver.inputs, saver.outputs, saver.grads,
        quantizers,
        iters=1000
    )

    print("Done!")