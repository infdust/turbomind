// #include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <random> 

__global__ void compute_max_val(const half* x, float* max_val, int group_size, int n_groups) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    int start_idx = group_id * group_size;
    int idx = start_idx + tid;
    // printf("hello");
    extern __shared__ float shared_max[];
    float local_max = 0.0f;

    if (idx < (group_id + 1) * group_size) {
        local_max = fabsf(__half2float(x[idx]));
    }
    // printf("hello");
    shared_max[tid] = local_max;
    __syncthreads();

    // 归约求最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_val[group_id] = fmaxf(shared_max[0], 1e-5f);
    }
}

__device__ __nv_fp4_e2m1 lookup_quantize(float x_scaled) {
    const float sorted_candidates[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float abs_x = fabsf(x_scaled);
    float closest = sorted_candidates[0];
    float min_diff = fabsf(abs_x - closest);

    for (int i = 1; i < 8; ++i) {
        float diff = fabsf(abs_x - sorted_candidates[i]);
        if (diff < min_diff) {
            min_diff = diff;
            closest = sorted_candidates[i];
        }
    }
    if (abs_x > sorted_candidates[7]) closest = sorted_candidates[7];
    // closest = 0;
    return __nv_fp4_e2m1(__float2half_rn(closest * copysignf(1.0f, x_scaled)));
}

// 比较查表法与 Intrinsic 法的比特差异
__global__ void compare_fp4_bits(const __half* input, 
                                uint8_t* lookup_result, 
                                uint8_t* intrinsic_result, 
                                int group_size, 
                                const float* max_vals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d: start\n", idx);

    int group_id = idx / group_size;
    // printf("hello");
    // printf("group_id %d", group_id);
    // printf("gridDim.x : %d", gridDim.x);
    // if (group_id >= gridDim.x) return;

    // 计算缩放因子
    float max_val = max_vals[group_id];
    float scale = max_val / 6.0f;

    // 缩放输入值
    float x_scaled = __half2float(input[idx]) / scale;

    // 查表法量化
    __nv_fp4_e2m1 q_lookup = lookup_quantize(x_scaled);
    // Intrinsic 法直接转换
    __nv_fp4_e2m1 q_intrinsic = __nv_fp4_e2m1(__float2half_rn(x_scaled));

    // 将 FP4 值按比特存储到结果中
    int byte_idx = idx / 2;
    int shift = (idx % 2) * 4;
    // printf("hello %d ", idx);
    lookup_result[byte_idx] |= (static_cast<uint8_t>(q_lookup) & 0xF) << shift;
    intrinsic_result[byte_idx] |= (static_cast<uint8_t>(q_intrinsic) & 0xF) << shift;
}


void check_bit_errors(const uint8_t* d_lookup, 
                     const uint8_t* d_intrinsic, 
                     int total_elements) {
    std::vector<uint8_t> h_lookup(total_elements / 2);
    std::vector<uint8_t> h_intrinsic(total_elements / 2);
    cudaMemcpy(h_lookup.data(), d_lookup, total_elements / 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_intrinsic.data(), d_intrinsic, total_elements / 2, cudaMemcpyDeviceToHost);

    // 打印 h_lookup 和 h_intrinsic 的内容
    printf("\n=== h_lookup (查表法) ===\n");
    for (size_t i = 0; i < h_lookup.size(); ++i) {
        printf("%02X ", h_lookup[i]);
        if ((i + 1) % 8 == 0) printf("\n");  // 每8个字节换行
    }

    printf("\n\n=== h_intrinsic (Intrinsic法) ===\n");
    for (size_t i = 0; i < h_intrinsic.size(); ++i) {
        printf("%02X ", h_intrinsic[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");

    int errors = 0;
    for (size_t i = 0; i < h_lookup.size(); ++i) {
        if (h_lookup[i] != h_intrinsic[i]) {
            // 检查每个 4-bit 单元
            uint8_t lookup_low = h_lookup[i] & 0xF;
            uint8_t intrinsic_low = h_intrinsic[i] & 0xF;
            uint8_t lookup_high = (h_lookup[i] >> 4) & 0xF;
            uint8_t intrinsic_high = (h_intrinsic[i] >> 4) & 0xF;

            if (lookup_low != intrinsic_low) errors++;
            if (lookup_high != intrinsic_high) errors++;
        }
    }
    printf("Bit-level errors: %d/%d (%.2f%%)\n", 
           errors, total_elements, 100.0f * errors / total_elements);
}


int main() {
    const int group_size = 16;
    const int n_groups = 16;
    const int total_elements = group_size * n_groups;

    // 初始化输入数据（正态分布）
    std::random_device rd;
    std::mt19937 gen(rd());
    
    const float mu = 0.0f;      // 均值
    const float sigma = 1.666f; // 标准差

    std::normal_distribution<float> dist(mu, sigma);
    std::vector<__half> h_input(total_elements);

    for (auto& val : h_input) {
        float num = dist(gen);
        val = __float2half_rn(num);
        // val = 100;
    }

    __half* d_input;
    float* d_max_vals;
    uint8_t* d_lookup_result, *d_intrinsic_result;
    cudaMalloc(&d_input, total_elements * sizeof(__half));
    cudaMalloc(&d_max_vals, n_groups * sizeof(float));
    cudaMalloc(&d_lookup_result, total_elements / 2);
    cudaMalloc(&d_intrinsic_result, total_elements / 2);
    
    cudaMemset(d_lookup_result, 0, total_elements / 2);
    cudaMemset(d_intrinsic_result, 0, total_elements / 2);

    cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);

    // 计算group-wise的最大值
    compute_max_val<<<n_groups, group_size, group_size * sizeof(float)>>>(d_input, d_max_vals, group_size, n_groups);
    cudaDeviceSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    // printf("hello");
    //分别计算查表法和intrinsic法的结果
    compare_fp4_bits<<<grid, block>>>(d_input, d_lookup_result, d_intrinsic_result, group_size, d_max_vals);
    cudaDeviceSynchronize();
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    //比较两者的bit-level error
    check_bit_errors(d_lookup_result, d_intrinsic_result, total_elements);
    // Bit-level errors: 0/256 (0.00%)
    cudaFree(d_input);
    cudaFree(d_max_vals);
    cudaFree(d_lookup_result);
    cudaFree(d_intrinsic_result);

    return 0;
}

// === h_lookup (查表法) ===
// 00 00 03 04 03 03 00 06 
// 06 00 06 00 00 00 00 00 
// 03 00 01 04 02 00 00 02 
// 02 01 03 00 00 04 04 00 
// 03 00 00 01 00 00 00 04 
// 00 04 00 00 01 00 00 00 
// 03 00 03 00 00 00 00 00 
// 00 00 02 00 00 00 04 00 
// 00 00 00 00 00 00 03 00 
// 00 00 00 04 01 06 00 01 
// 02 01 00 00 06 00 00 00 
// 00 02 00 00 01 00 00 00 
// 03 00 00 00 00 00 00 02 
// 00 04 04 00 06 00 00 00 
// 02 00 01 00 00 03 06 00 
// 00 00 00 06 06 02 01 00 


// === h_intrinsic (Intrinsic法) ===
// 00 00 03 04 03 03 00 06 
// 06 00 06 00 00 00 00 00 
// 03 00 01 04 02 00 00 02 
// 02 01 03 00 00 04 04 00 
// 03 00 00 01 00 00 00 04 
// 00 04 00 00 01 00 00 00 
// 03 00 03 00 00 00 00 00 
// 00 00 02 00 00 00 04 00 
// 00 00 00 00 00 00 03 00 
// 00 00 00 04 01 06 00 01 
// 02 01 00 00 06 00 00 00 
// 00 02 00 00 01 00 00 00 
// 03 00 00 00 00 00 00 02 
// 00 04 04 00 06 00 00 00 
// 02 00 01 00 00 03 06 00 
// 00 00 00 06 06 02 01 00 

// Bit-level errors: 0/256 (0.00%)