#include <cuda_runtime.h>
#include <torch/extension.h>

template<int BM, int BN, int BK, int TM, int TN_PHY>
__global__ void sgemm2D_kernel(int M, int K, int N, float* A, int* B, float* C, float* scale, float* zero_point)
{
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    constexpr int TN_LOG = TN_PHY * 8;

    int threadRow = threadIdx.x / (BN / TN_LOG);
    int threadCol = threadIdx.x % (BN / TN_LOG);

    __shared__ float As[BM * BK];
    __shared__ int Bs[BK * (BN / 8)];

    // 128 * 8 = 1024 , 256 threads ,
    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);

    // (128 / 8) * 8 = 128
    int innerRowB = threadIdx.x / (BN / 8 / 4);
    int innerColB = threadIdx.x % (BN / 8 / 4);

    A += cRow * BM * K;
    B += cCol * (BN / 8);
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM * TN_LOG] = {0.0};
    float regM[TM] = {0.0};
    int regN[TN_PHY] = {0};

    float my_scales[TN_LOG];
    float my_zeros[TN_LOG];

    for (int i = 0; i < TN_LOG; ++i) {
        int globalN = cCol * BN + threadCol * TN_LOG + i;
        if (globalN < N) {
            my_scales[i] = scale[globalN];
            my_zeros[i]  = zero_point[globalN];
        }
        else {
            my_scales[i] = 1.0f;
            my_zeros[i] = 0.0f;
        }
    }

    for (int bk = 0; bk < K; bk += BK) {

        int globalM = cRow * BM + innerRowA;
        int globalK = bk + innerColA * 4;

        float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

        if (globalM < M) {
            if (globalK + 4 <= K) {
                tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
            }
            else {
                float* ptr = &A[innerRowA * K + innerColA * 4];
                if (globalK + 0 < K) tmp.x = ptr[0];
                if (globalK + 1 < K) tmp.y = ptr[1];
                if (globalK + 2 < K) tmp.z = ptr[2];
                if (globalK + 3 < K) tmp.w = ptr[3];
            }
        }

        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // 8 * 4 = 32 개 불러옴
        if (threadIdx.x < BK * (BN / 8) / 4) {

            if (bk + innerRowB < K) {
                reinterpret_cast<int4 *>(&Bs[innerRowB * (BN / 8) + innerColB * 4])[0] =
                reinterpret_cast<int4 *>(&B[innerRowB * (N / 8) + innerColB * 4])[0];
            }
            else {
                reinterpret_cast<int4 *>(&Bs[innerRowB * (BN / 8) + innerColB * 4])[0] = {0, 0, 0, 0};
            }
        }
        __syncthreads();

        A += BK;
        B += BK * (N / 8);

        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i){
                regM[i] = As[dot * BM + threadRow * TM + i];
            }
            for (int i = 0; i < TN_PHY; ++i) {
                regN[i] = Bs[dot * (BN / 8) + threadCol * TN_PHY + i];
            }

            for (int i = 0; i < TN_PHY; ++i) {
                int packed_val = regN[i];

                for (int subN = 0; subN < 8; ++subN) {
                    int int4_val = (packed_val >> (subN * 4)) & 0xF;
                    float real_val = (float(int4_val) - my_zeros[i * 8 + subN]) * my_scales[i * 8 + subN];

                    for (int m = 0; m < TM; ++m) {
                        int resNidx = i * 8 + subN;

                        threadResults[m * TN_LOG + resNidx] += regM[m] * real_val;
                    }
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN_LOG; resIdxN += 4) {
            int globalRowC = cRow * BM + threadRow * TM + resIdxM;
            int globalColC = cCol * BN + threadCol * TN_LOG + resIdxN;

            if (globalRowC < M) {
                float4 tmp;
                tmp.x = threadResults[resIdxM * TN_LOG + resIdxN];
                tmp.y = threadResults[resIdxM * TN_LOG + resIdxN + 1];
                tmp.z = threadResults[resIdxM * TN_LOG + resIdxN + 2];
                tmp.w = threadResults[resIdxM * TN_LOG + resIdxN + 3];

                if (globalColC + 4 <= N) {
                    reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN_LOG + resIdxN])[0] = tmp;
                }
                else {
                    float* ptr = &C[(threadRow * TM + resIdxM) * N + threadCol * TN_LOG + resIdxN];
                    if (globalColC + 0 < N) ptr[0] = tmp.x;
                    if (globalColC + 1 < N) ptr[1] = tmp.y;
                    if (globalColC + 2 < N) ptr[2] = tmp.z;
                    if (globalColC + 3 < N) ptr[3] = tmp.w;
                }
            }
        }
    }
}

void sgemm_int4_cuda(torch::Tensor A, torch::Tensor B_packed, torch::Tensor C, torch::Tensor scale, torch::Tensor zero_point) {

    // [Safety Checks]
    // 커널이 Vectorized Load를 수행하므로 차원과 정렬이 안 맞으면 SegFault 발생.
    // 따라서 C++ 단에서 엄격하게 검사합니다.

    int M = A.size(0);
    int K = A.size(1);
    int N = C.size(1);

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;

    // 1. 배수 조건 검사
    TORCH_CHECK(N % BN == 0, "N must be a multiple of 128");

    // 2. Vectorization 조건 검사
    // A inner dim (K) must be multiple of 4 (float4)
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4 for float4 loading");
    // B packed inner dim (N/8) must be multiple of 4 (int4 loading) -> N multiple of 32
    TORCH_CHECK(N % 32 == 0, "N must be multiple of 32 for int4 loading");

    // 3. Contiguity 검사
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B_packed.is_contiguous(), "B_packed must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(zero_point.is_contiguous(), "zero_point must be contiguous");

    dim3 blockDim(256);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm2D_kernel<BM, BN, BK, 8, 1><<<gridDim, blockDim>>>(
        M, K, N,
        A.data_ptr<float>(),
        B_packed.data_ptr<int>(),
        C.data_ptr<float>(),
        scale.data_ptr<float>(),
        zero_point.data_ptr<float>()
    );
}