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
template<>
__device__ __forceinline__ double warp_reduce_max(double value) {
  #pragma unroll
  for(size_t i = WARP_SZIE >> 1;i > 0;i >>= 1) {
    value = fmax(value, __shfl_xor_sync(0xffffffff, value, i));
  }
  return value;
}
__device__ __forceinline__ double safe_exp(double x) {
    double diff = x;
    
    // 当差异很小时，使用更高精度计算
    if (fabs(diff) < 1e-4) {
        // 使用 exp(x) = 1 + x + x²/2 + x³/6 + ... 更高阶
        double result = 1.0;
        double term = 1.0;
        for (int i = 1; i <= 8; i++) {
            term *= diff / i;
            result += term;
        }
        return result;
    }
    
    // 正常情况
    if (diff < -80.0) return 0.0;
    if (diff > 80.0) return exp(80.0);
    return exp(diff);
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


template<typename T, const int Br = 16, const int Bc = 32, const int TM = 1, const int TN = 2>
__global__ void flash_attn_kernel(T *q, T *k, T *v, T *o, 
                          const int q_len, const int kv_len, const int kv_heads, const int dim, const bool is_causal, const float scale) {
  const int BM = Br * TM;
  const int BN = Bc * TN;
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
  q += batch_id * q_len * q_stride + head_id * dim + bx * BM * q_stride;
  k += batch_id * kv_len * kv_stride + kv_head_id * dim;
  v += batch_id * kv_len * kv_stride + kv_head_id * dim;
  o += batch_id * q_len * q_stride + head_id * dim + bx * BM * q_stride;
  
  const int Tc = CEIL(kv_len, BN);
  
  
  extern __shared__ char smem[];
  
  // shared_mem shape: for bcf 
  // q o: Br * dim
  // k v: dim * Bc
  float *s_q = (float*)smem;
  float *s_k = s_q + BM * dim;
  float *s_v = s_k + BN * dim;
  float *s_o = s_v + BN * dim;
  float *s_m = s_o + BM * dim;
  float *s_l = s_m + BM;
  
  #pragma unroll
  for(size_t i = tid;i < BM;i += block_size) {
    s_m[i] = -__FLT_MAX__;
    s_l[i] = 0;
  }
  int q_acc_len = bx * BM;
  int kv_acc_len = 0;

  #pragma unroll
  for(size_t i = tid;i < BM * dim;i += block_size) {
    int x = i % dim;
    int y = i / dim;
    s_q[i] = ((q_acc_len + y) < q_len) ? static_cast<float>(q[y * q_stride + x]) : float(0);
  }
  #pragma unroll
  for(size_t c = 0;c < Tc;++c) {
    #pragma unroll
    for(size_t i = tid;i < BN * dim;i += block_size) {
      int x = i % dim;
      int y = i / dim;
      s_k[x * BN + y] = ((kv_acc_len + y) < kv_len) ? static_cast<float>(k[y * kv_stride + x]) : float(0);
      s_v[x * BN + y] = ((kv_acc_len + y) < kv_len) ? static_cast<float>(v[y * kv_stride + x]) : float(0);
    }
    __syncthreads();
    // sum[0]: ty tx 、 sum[1]: ty tx + Bc
    float sum[TN] = {0.f};
    #pragma unroll
    for(size_t i = 0;i < dim;++i) {
        float tmp = s_q[ty * dim + i];
        #pragma unroll
        for(size_t j = 0;j < TN;j++) {
            sum[j] += tmp * s_k[i * BN + tx + j * Bc];
        }
    }
    float m_sum[TN];
    float m_now = -__FLT_MAX__;
    float l_now = 0.f;
    float m_pre = s_m[ty];
    float l_pre = s_l[ty];
    float m = m_pre;
    float l = l_pre;

    #pragma unroll
    for(size_t i = 0;i < TN;i++) {
        sum[i] *= scale;
        m_sum[i] = (((q_acc_len + ty < q_len) && (kv_acc_len + tx + i * Bc < kv_len)) && (!is_causal || q_acc_len + ty >= kv_acc_len + tx + i * Bc)) ? sum[i] : -__FLT_MAX__;
        m_now = fmaxf(m_now, m_sum[i]);
        sum[i] = (((q_acc_len + ty < q_len) && (kv_acc_len + tx + i * Bc < kv_len)) && (!is_causal || q_acc_len + ty >= kv_acc_len + tx + i * Bc)) ? __expf(sum[i]) : 0.f;
        l_now += sum[i];
    }
    m_now = warp_reduce_max<float>(m_now);
    float expf_m_now = __expf(-m_now);
    l_now *= expf_m_now;
    l_now = warp_reduce_sum<float>(l_now);
    float p_now[TN];
    #pragma unroll
    for(size_t i = 0;i < TN;i++) {
        p_now[i] = sum[i] * expf_m_now;
    }
    m = fmaxf(m, m_now);
    l = l_pre * __expf(m_pre - m) + l_now * __expf(m_now - m);
    s_m[ty] = m;
    s_l[ty] = l;

    float exp_mprem = __expf(m_pre - m);
    float exp_mnowm = __expf(m_now - m);
    #pragma unroll
    for(size_t j = 0;j < dim;++j) {
        float value = 0.f;
        for(size_t i = 0;i < TN;i++) {
            value += p_now[i] * s_v[j * BN + tx + Bc * i];
        }
        value = warp_reduce_sum<float>(value);
        if(laneid == 0)
            s_o[ty * dim + j] = (q_acc_len + ty < q_len) ? (s_o[ty * dim + j] * exp_mprem * l_pre + value * exp_mnowm) / l : 0.f;
    }

    k += BN * kv_stride;
    v += BN * kv_stride;
    kv_acc_len += BN;
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
  if(is_causal) {
    //printf("is_causal\n");
    //return ;
  }
  // TODO: Implement the flash attention functio
  //printf("batch_size : %d\n", batch_size);
  //printf("target_seq_len : %d\n", target_seq_len);
  //printf("src_seq_len : %d\n", src_seq_len);
  //printf("query_heads : %d\n", query_heads);
  //printf("kv_heads : %d\n", kv_heads);
  printf("head_dim : %d\n", head_dim);
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
  // (max_sram-16*2) / 2 : 6128
  // (max_sram-32*2) / 2 : 6112
  // TM = (6128 / head_dim - 16) / 32
  // TM = (6112 / head_dim - 32) / 32
  switch (head_dim)
  {
  case 1:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 8;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 2:
    // TM = (6128 / 2 - 16) / 32 = 95
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 8;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 4:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 8;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 8:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 8;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 16:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 8;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 32:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 4;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 64:
    {
      constexpr int Br = 16;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 2;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br), query_heads, batch_size);
      int sram_bytes = ((Br * TM + Bc * TN) * head_dim * 2 + Br * TM * 2) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  default:
    printf("head_dim : %d\n", head_dim);
    break;
  }
    cudaMemcpy(h_o.data(), d_o, qo_bytes, cudaMemcpyDeviceToHost);
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
