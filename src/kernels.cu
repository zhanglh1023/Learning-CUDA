#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#define CEIL(N, M) (((N) + (M) - 1) / (M))
#define WARP_SZIE 32

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
  #pragma unroll
  for(size_t i = WARP_SZIE >> 1;i > 0;i >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, i);
  }
  return value;
}
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T value) {
  #pragma unroll
  for(size_t i = WARP_SZIE >> 1;i > 0;i >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, i));
  }
  return value;
}
/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template<typename T>
__global__ void trace_kernel(T *input, T *output, int cols, int n, const int STRIDE = 1024, const int NUM_PER_WARP = 4) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  T value = 0;
  #pragma unroll
  for(size_t i = 0;i < NUM_PER_WARP;i++) {
    size_t x = idx + i * STRIDE;
    if(x < n) value += input[x * cols + x];
  }
  value = warp_reduce_sum<T>(value);
  const int warpid = tid / 32;
  const int laneid = tid % 32;
  __shared__ T smem[32];
  if(laneid == 0) {
    smem[warpid] = value;
  }
  __syncthreads();
  if(warpid == 0) {
    value = smem[laneid];
    value = warp_reduce_sum<T>(value);
    if(laneid == 0) {atomicAdd(output, value);}
  }
}
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t n = rows * cols;
  size_t bytes = n * sizeof(T);
  T *input_d, *output_d;
  cudaMalloc((void**)(&input_d), bytes);
  cudaMalloc((void**)(&output_d), sizeof(T));
  cudaMemcpy(input_d, h_input.data(), bytes, cudaMemcpyHostToDevice);
  T *output_h = (T*)malloc(sizeof(T));
  *output_h = 0;
  cudaMemcpy(output_d, output_h, sizeof(T), cudaMemcpyHostToDevice);
  size_t diagonal = min(rows, cols);
  dim3 block(256);
  dim3 grid(1);
  if(diagonal <= 4096) {
    const int STRIDE = 256;
    const int NUM_PER_WARP = CEIL(diagonal, 256);
    trace_kernel<T><<<grid, block>>>(input_d, output_d, cols, diagonal, STRIDE, NUM_PER_WARP);
  } else {
    grid.x = (CEIL(CEIL(diagonal, 16), 256));
    const int NUM_PER_WARP = 16;
    const int STRIDE = 256 * grid.x;
    trace_kernel<T><<<grid, block>>>(input_d, output_d, cols, diagonal, STRIDE, NUM_PER_WARP);
  }
  cudaMemcpy(output_h, output_d, sizeof(T), cudaMemcpyDeviceToHost);
  return T((*output_h));
}


template<typename T, const int Br = 16, const int Bc = 32>
__global__ void flash_attn_kernel(T *q, T *k, T *v, T *o, 
                          const int q_len, const int kv_len, const int kv_heads, const int dim, const bool is_causal, const float scale) {
  const int bx = blockIdx.x;
  const int block_size = blockDim.x;
  const int head_id = blockIdx.y;
  const int kv_head_id = head_id / (gridDim.y / kv_heads);
  const int batch_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int tx = tid % Bc;
  const int ty = tid / Bc;
  const int laneid = tid % WARP_SZIE;
  const int q_stride = gridDim.y * dim;
  const int kv_stride = kv_heads * dim;
  q += batch_id * q_len * q_stride + head_id * dim + bx * Br * q_stride;
  k += batch_id * kv_len * kv_stride + kv_head_id * dim;
  v += batch_id * kv_len * kv_stride + kv_head_id * dim;
  o += batch_id * q_len * q_stride + head_id * dim + bx * Br * q_stride;
  
  const int Tc = CEIL(kv_len, Bc);
  
  
  extern __shared__ char smem[];
  
  float *s_q = (float*)smem;
  float *s_k = s_q + Br * dim;
  float *s_v = s_k + Bc * dim;
  float *s_o = s_v + Bc * dim;
  float *s_m = s_o + Br * dim;
  float *s_l = s_m + Br;
  
  #pragma unroll
  for(size_t i = tid;i < Br;i += block_size) {
    s_m[i] = -__FLT_MAX__;
    s_l[i] = 0;
  }
  int q_acc_len = bx * Br;
  int kv_acc_len = 0;

  #pragma unroll
  for(size_t i = tid;i < Br * dim;i += block_size) {
    int x = i % dim;
    int y = i / dim;
    s_q[i] = ((q_acc_len + y) < q_len) ? static_cast<float>(q[y * q_stride + x]) : float(0);
  }
  #pragma unroll
  for(size_t c = 0;c < Tc;++c) {
    #pragma unroll
    for(size_t i = tid;i < Bc * dim;i += block_size) {
      int x = i % dim;
      int y = i / dim;
      s_k[i] = ((kv_acc_len + y) < kv_len) ? static_cast<float>(k[y * kv_stride + x]) : float(0);
      s_v[i] = ((kv_acc_len + y) < kv_len) ? static_cast<float>(v[y * kv_stride + x]) : float(0);
    }
    __syncthreads();

    float sum = 0.f;
    #pragma unroll
    for(size_t i = 0;i < dim;++i) {
      sum += s_q[ty * dim + i] * s_k[tx * dim + i];
    }
    sum *= scale;
    float m_now = (((q_acc_len + ty < q_len) && (kv_acc_len + tx < kv_len)) && (!is_causal || q_acc_len + ty >= kv_acc_len + tx)) ? sum : -__FLT_MAX__;
    m_now = warp_reduce_max<float>(m_now);
    sum = (((q_acc_len + ty < q_len) && (kv_acc_len + tx < kv_len)) && (!is_causal || q_acc_len + ty >= kv_acc_len + tx)) ? __expf(sum - m_now) : 0.f;
    float l_now = sum;
    l_now = warp_reduce_sum<float>(l_now);
    
    float m_pre = s_m[ty];
    float l_pre = s_l[ty];

    float m = fmaxf(m_pre, m_now);
    float l = l_pre * exp(m_pre - m) + l_now * exp(m_now - m);
    s_m[ty] = m;
    s_l[ty] = l;
    #pragma unroll
    for(size_t i = 0;i < dim;++i) {
      float value = sum * s_v[tx * dim + i];
      value = warp_reduce_sum<float>(value);
      if(laneid == 0)
        s_o[ty * dim + i] = (q_acc_len + ty < q_len) ? (s_o[ty * dim + i] * exp(m_pre - m) * l_pre + value * exp(m_now - m)) / l : 0.f;
    }
    // e^(x-m) / l * v
    k += Bc * kv_stride;
    v += Bc * kv_stride;
    kv_acc_len += Bc;
    __syncthreads();
  }
  //if(laneid == 0) {
    //printf("m[%d]: %.2f\n", q_acc_len + ty, s_m[ty]);
    //printf("l[%d]: %.2f\n", q_acc_len + ty, s_l[ty]);
  //}
  #pragma unroll
  for(size_t i = tid;i < Br * dim;i += block_size) {
    int x = i % dim;
    int y = i / dim;
    if(q_acc_len + y < q_len) {
        //printf("o[%d]: %.2f\n", q_acc_len + y, s_o[y * dim + x]);
        o[y * q_stride + x] = static_cast<T>(s_o[y * dim + x]);
    }
  }
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {     
  //if(is_causal) return ;  
  // TODO: Implement the flash attention functio
  //printf("batch_size : %d\n", batch_size);
  //printf("target_seq_len : %d\n", target_seq_len);
  //printf("src_seq_len : %d\n", src_seq_len);
  //printf("query_heads : %d\n", query_heads);
  //printf("kv_heads : %d\n", kv_heads);
  //printf("head_dim : %d\n", head_dim);
  size_t qo_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t qo_bytes = qo_size * sizeof(T);
  size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t kv_bytes = kv_size * sizeof(T);
  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc((void**)(&d_q), qo_bytes);
  cudaMalloc((void**)(&d_k), kv_bytes);
  cudaMalloc((void**)(&d_v), kv_bytes);
  cudaMalloc((void**)(&d_o), qo_bytes);
  cudaMemcpy(d_q, h_q.data(), qo_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);
  
  int max_sram_bytes;
  cudaDeviceGetAttribute(&max_sram_bytes, cudaDevAttrMaxSharedMemoryPerBlock, 0);//12288 floats 
  size_t max_sram_size = max_sram_bytes / sizeof(T);
  //printf("max_sram_size : %d\n", max_sram_size);
  if(head_dim <= 128) {
    constexpr int Br = 16;
    constexpr int Bc = 32;
    dim3 block(512);
    dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
    size_t sram_size = (Br + Bc) * head_dim * 2 + Br;
    int sram_bytes = sram_size * sizeof(float) + Br * sizeof(float);
    sram_bytes = min(sram_bytes, max_sram_bytes);
    flash_attn_kernel<T, Br, Bc><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, 1.0 / sqrt(head_dim));
    cudaMemcpy(h_o.data(), d_o, qo_bytes, cudaMemcpyDeviceToHost);
  }
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
