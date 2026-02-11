import torch
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def block_reconstruction(block, cali_inputs, cali_outputs, cali_grads, quantizers, batch_size=64, iters=1000):

    # 학습 파라미터 추출 및 adaround 설정
    params = []
    for q in quantizers:
        params.append(q.alpha)
        q.soft_targets = True

    optimizer = Adam(params, lr=1e-3)

    dataset = TensorDataset(cali_inputs.cpu(), cali_outputs.cpu(), cali_grads.cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    beta_scheduler = np.linspace(20, 2, iters)

    for i in range(iters):
        beta = beta_scheduler[i]

        for x_batch, y_batch, g_batch in loader:
            # 여기서 데이터를 GPU로 이동하여 메모리 절약
            cur_input = x_batch.to(device) # Input
            cur_target = y_batch.to(device) # Output (Target)
            cur_grad = g_batch.to(device) # Fisher (Weight)

            optimizer.zero_grad()

            # Forward (Soft Quantization)
            out_quant = block(cur_input)

            # Fisher Loss 계산
            # sum( (output_diff * grad)^2 )
            delta = out_quant - cur_target

            loss_rec = (delta * cur_grad).pow(2).sum()

            # 데이터 개수로 정규화
            loss_rec = loss_rec / batch_size

            # Regularization Loss
            loss_reg = 0

            for q in quantizers:
                soft_val = q.rectified_sigmoid()
                reg_term = 1.0 - (2 * soft_val - 1).abs().pow(beta)
                loss_reg += reg_term.sum()

            # Total Loss
            total_loss = loss_rec + 1e-4 * loss_reg

            total_loss.backward()
            optimizer.step()

        if i % 200 == 0:
            print(f"Iter {i}: Total {total_loss.item():.4f} (Rec {loss_rec.item():.10f})")


    # 학습 종료 후 Hard Mode로 전환
    for q in quantizers:
        q.soft_targets = False