#include <stdio.h>
#include <iostream>
#include <cuda_fp8.h>

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// For making the host code a bit easier to read
// Convert 32-bit float to cuda built-in Sign-1-bit Exponent-4-bits Mantissa-3-bits
// Would call this fp8_e4m3(), but e4m3 is the only one we're using here
// Doc: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__E4M3__STRUCT.html#group__CUDA__MATH__FP8__E4M3__STRUCT
// Usage example: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtFp8Matmul/main.cpp
__host__ __device__
__nv_fp8_e4m3 fp8(float x) {
    return __nv_fp8_e4m3(x);
}

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void quant_kernel(__nv_fp8_e4m3* params_memory, __nv_fp8_e4m3* grads_memory, __nv_fp8_e4m3* m_memory, __nv_fp8_e4m3* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, 
                              float eps, float weight_decay) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters) return;                // guard
    
    __nv_fp8_e4m3 grad_8 = grads_memory[i];         // load fp8             { rest of loads will
    float grad_32 = float(grad_8);                  // convert to fp32      { follow this pattern

    __nv_fp8_e4m3 m_8 = m_memory[i];
    __nv_fp8_e4m3 v_8 = v_memory[i];

    float m_32 = float(m_8);
    float v_32 = float(v_8);
    
    m_32 = lerp(grad_32, m_32, beta1);              // update the first moment (momentum)
    m_memory[i] = fp8(m_32);

    v_32 = lerp(grad_32 * grad_32, v_32, beta2);    // update the second moment (RMSprop)
    v_memory[i] = fp8(v_32);

    m_32 /= beta1_correction;  // m_hat
    v_32 /= beta2_correction;  // v_hat

    __nv_fp8_e4m3 param_8 = params_memory[i];
    float param_32 = float(param_8);
    float update_32 = param_32 - (learning_rate * (m_32 / (sqrtf(v_32) + eps) + weight_decay * param_32));

    __nv_fp8_e4m3 update_8 = fp8(update_32);
    params_memory[i] = update_8;
}

int main() {
    long num_parameters = 32;

    __nv_fp8_e4m3 params_memory[num_parameters] = {fp8(0.08156189322471619), fp8(0.3785102963447571), fp8(0.23286126554012299), fp8(0.9647358655929565), fp8(0.4282546639442444), fp8(0.7482216954231262), fp8(0.903114378452301), fp8(0.3822559118270874), fp8(0.3563106954097748), fp8(0.39088377356529236), fp8(0.2661018669605255), fp8(0.45732927322387695), fp8(0.356448769569397), fp8(0.5366447567939758), fp8(0.9373241662979126), fp8(0.2961907982826233), fp8(0.8248701095581055), fp8(0.6990491151809692), fp8(0.002520027570426464), fp8(0.9591174125671387), fp8(0.9756536483764648), fp8(0.493215948343277), fp8(0.678508996963501), fp8(0.8220535516738892), fp8(0.3433856666088104), fp8(0.012765476480126381), fp8(0.9194097518920898), fp8(0.7243597507476807), fp8(0.30336636304855347), fp8(0.8506981134414673), fp8(0.9834323525428772), fp8(0.3326418697834015)};
    __nv_fp8_e4m3 * d_params_memory;
    cudaMalloc(&d_params_memory, 4 * sizeof(__nv_fp8_e4m3));
    cudaMemcpy(d_params_memory, params_memory, 4 * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());

    // fake gradients
    __nv_fp8_e4m3 grads_memory[num_parameters] = {fp8(0.7289007306098938), fp8(0.30462440848350525), fp8(0.9082905054092407), fp8(0.31704971194267273), fp8(0.19741280376911163), fp8(0.5811731815338135), fp8(0.9425305724143982), fp8(0.43781372904777527), fp8(0.09683270007371902), fp8(0.12920717895030975), fp8(0.8269669413566589), fp8(0.7294973134994507), fp8(0.9390449523925781), fp8(0.155783012509346), fp8(0.5775147676467896), fp8(0.6951613426208496), fp8(0.49144434928894043), fp8(0.16329661011695862), fp8(0.2072339653968811), fp8(0.27448904514312744), fp8(0.43389183282852173), fp8(0.8969299793243408), fp8(0.6707720160484314), fp8(0.3562951683998108), fp8(0.9982314109802246), fp8(0.4646815061569214), fp8(0.560585081577301), fp8(0.9774811863899231), fp8(0.6622148752212524), fp8(0.19557878375053406), fp8(0.23262782394886017), fp8(0.802483081817627)};
    __nv_fp8_e4m3 * d_grads_memory;
    cudaMalloc(&d_grads_memory, sizeof(grads_memory));
    cudaMemcpy(d_grads_memory, grads_memory, sizeof(grads_memory), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());

    __nv_fp8_e4m3 m_memory[num_parameters] = {fp8(0)};
    __nv_fp8_e4m3 * d_m_memory;
    cudaMalloc(&d_m_memory, sizeof(m_memory));
    cudaMemcpy(d_m_memory, m_memory, sizeof(m_memory), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());
    
    __nv_fp8_e4m3 v_memory[num_parameters] = {fp8(0)};
    __nv_fp8_e4m3 * d_v_memory;
    cudaMalloc(&d_v_memory, sizeof(v_memory));
    cudaMemcpy(d_v_memory, v_memory, sizeof(v_memory), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());
    
    // standard choices / https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    int t = 1;
    float beta1_correction = 1.0f - powf(beta1, t);;
    float beta2_correction = 1.0f - powf(beta2, t);;
    float eps = 1e-08;
    float weight_decay = 0.01;
    
    quant_kernel<<<1, num_parameters>>>(d_params_memory, d_grads_memory, d_m_memory, d_v_memory, num_parameters,
                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());

    cudaMemcpy(params_memory, d_params_memory, 4 * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
    cudaCheck(cudaGetLastError());

    // TODO - Free device memory
    
    for (int i = 0; i < num_parameters; i++) {
        std::cout << "Updated parameters: " << float(params_memory[i]) << "\n";
    }
}
