# BRECQ Implementation & W4A32 Custom CUDA Kernel Acceleration

본 프로젝트는 거대 모델의 엣지 디바이스 배포를 위해 **BRECQ(Block Reconstruction Quantization)** 알고리즘을 직접 구현하고, PyTorch에서 지원하지 않는 INT4 연산을 가속하기 위해 **Custom CUDA Kernel**을 설계한 하드웨어 친화적 최적화 연구입니다.  

## 1. 문제 정의 (Problem Statement)

- **자원 제약적 환경**: 거대 모델은 메모리와 연산 비용이 높아 드론과 같은 엣지 디바이스 배포에 한계가 있음.  
    
- **추론 엔진 부재**: PyTorch는 기본적으로 INT4 연산을 지원하지 않으므로, 성능 이득을 위해서는 직접적인 CUDA 커널 구현이 필수적임.  
    

## 2. 핵심 해결 과정 (Solution)

### Quantization Algorithm (BRECQ)

정확도 손실을 최소화하기 위한 **PTQ(Post-Training Quantization)** 기법을 적용했습니다.  

- **BN Folding**: Inference 속도 향상을 위해 Conv와 Batch Norm 레이어를 융합함.  
    
- **Fisher Information Loss**: 단순 MSE가 아닌 출력의 Gradient(Hessian 근사)를 고려하여 가중치 최적화를 수행함.  
    
- **AdaRound**: 학습 가능한 파라미터를 도입하여 최적의 가중치 정수 값을 탐색함.  
    

###  System Optimization (CUDA Kernel)

저수준 하드웨어 최적화를 통해 연산 효율을 극대화했습니다.  

- **W4A32 Strategy**: 가중치는 4-bit로 압축하여 메모리 대역폭을 절약하고, 연산은 32-bit로 수행하여 정확도를 방어함.  
    
- **Bit Packing**: 32-bit int 하나에 8개의 4-bit 가중치를 패킹(Packing)하여 저장 효율을 높임.  
    
- **Shared Memory Tiling**: Global Memory 접근을 최소화하기 위해 **2D Block Tiling** 기법을 적용함.  
    

## 3. 주요 성과 (Result)

- **정확도(Accuracy)**: FP32(67.79%) 대비 INT4(65.38%)로 **약 2.4%의 Drop**만으로 **8배의 가중치 압축** 성공.  
    
- **프로파일링 분석**:
    
    - **병목 구간 이동**: Base 모델(CPU Bound, 57%) 대비 구현 모델은 GPU 연산 비중이 50%까지 상승하며 **GPU Bound** 상태로 전환됨을 확인.  
        
    - **Compute Throughput**: **71.25%** 달성 (연산 자원의 효율적 사용 증명).  
        

## 4. 이슈 해결 (Troubleshooting)

- **메모리 정렬 문제**: `unfold` 연산 후 비연속적인 메모리 레이아웃 문제를 `.contiguous()` 호출로 해결하여 **Vectorized Load** 시 데이터 오류 방지.  
    
- **패킹 경계 처리**: 가중치 패킹 시 발생하는 Boundary Check 복잡성을 해결하기 위해 출력 채널을 **block size(128)** 단위로 패딩 처리함.  
    

---

**Author**: 박상기 (SANGGI PARK)  

**Contact**: remind326@gmail.com