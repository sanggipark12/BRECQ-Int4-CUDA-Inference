#include <cuda_runtime.h>

template<int BM, int BN, int BK, int TM, int TN_PHY>
__global__ void sgemm2D(float* Input, int* B, float* C, float* scale, float* zero_point,
    int batch,
    int In_H, int In_W, int In_C,
    int Out_H, int Out_W,
    int K_H, int K_W,
    int Pad_H, int Pad_W,
    int Stride_H, int Stride_W,
    int Dilation_H, int Dilation_W)
{
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int M = batch * Out_H * Out_W;
    int N = gridDim.x * BN;
    int K = In_C * K_H * K_W;

    constexpr int TN_LOG = TN_PHY * 8;

    int threadRow = threadIdx.x / (BN / TN_LOG);
    int threadCol = threadIdx.x % (BN / TN_LOG);

    __shared__ float As[BM * BK];
    __shared__ int Bs[BK * (BN / 8)];

    B += cCol * (BN / 8);
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM * TN_LOG] = {0.0};
    float regM[TM] = {0.0};
    int regN[TN_PHY] = {0};

    float sumA[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {

        for (int idx = threadIdx.x; idx < BK * BM; idx += blockDim.x) {
            int r = idx % BM;
            int c = idx / BM;

            int globalM = cRow * BM + r;
            int curr_k = bk + c;

            float val = 0.0f;
            if (curr_k < K && globalM < M) {
                int area_out = Out_H * Out_W;
                int batch_idx = globalM / area_out;
                int pixel_rem = globalM % area_out;
                int oh = pixel_rem / Out_W;
                int ow = pixel_rem % Out_W;

                int kc = curr_k % K_W;
                int k_rem = curr_k / K_W;
                int kh = k_rem % K_H;
                int ic = k_rem / K_H;

                int ih = oh * Stride_H - Pad_H + kh * Dilation_H;
                int iw = ow * Stride_W - Pad_W + kc * Dilation_W;

                if (ih >= 0 && ih < In_H && iw >= 0 && iw < In_W) {
                    long long input_idx =
                        (long long)batch_idx * (In_C * In_H * In_W) +
                        (long long)ic * (In_H * In_W) +
                        (long long)ih * In_W + iw;
                    val = Input[input_idx];
                }
            }
            As[c * BM + r] = val;
        }

        int num_int4 = (BK * BN / 8) / 4;
        for (int idx = threadIdx.x; idx < num_int4; idx += blockDim.x) {
            int r = idx / ((BN / 8) / 4);
            int c_int4 = idx % ((BN / 8) / 4);
            int c = c_int4 * 4;

            if (bk + r < K) {
                reinterpret_cast<int4*>(&Bs[r * (BN / 8) + c])[0] =
                    reinterpret_cast<int4*>(&B[r * (N / 8) + c])[0];
            } else {
                Bs[r * (BN / 8) + c + 0] = 0;
                Bs[r * (BN / 8) + c + 1] = 0;
                Bs[r * (BN / 8) + c + 2] = 0;
                Bs[r * (BN / 8) + c + 3] = 0;
            }
        }
        __syncthreads();

        B += BK * (N / 8);

        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i){
                float a_val = As[dot * BM + threadRow * TM + i];
                regM[i] = a_val;
                sumA[i] += a_val; 
            }
            for (int i = 0; i < TN_PHY; ++i) {
                regN[i] = Bs[dot * (BN / 8) + threadCol * TN_PHY + i];
            }

            for (int i = 0; i < TN_PHY; ++i) {
                int packed_val = regN[i];

                for (int subN = 0; subN < 8; ++subN) {
                    int int4_val = (packed_val >> (subN * 4)) & 0xF;
                    float w_val = float(int4_val); 

                    for (int m = 0; m < TM; ++m) {
                        int resNidx = i * 8 + subN;
                        threadResults[m * TN_LOG + resNidx] += regM[m] * w_val;
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int resIdxN = 0; resIdxN < TN_LOG; ++resIdxN) {
        int globalN = cCol * BN + threadCol * TN_LOG + resIdxN;
        float s = 1.0f;
        float z = 0.0f;

        // 메모리에서 여기서 불러옴 (글로벌 메모리 병목 회피)
        if (globalN < N) {
            s = scale[globalN];
            z = zero_point[globalN];
        }

        for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
            // C = (A*W - sum(A)*Z) * S
            threadResults[resIdxM * TN_LOG + resIdxN] =
                (threadResults[resIdxM * TN_LOG + resIdxN] - sumA[resIdxM] * z) * s;
        }
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