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

  T value = (T)(0);
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
    value = laneid < 8 ? smem[laneid] : T(0);
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
  *output_h = (T)0;
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
  cudaDeviceSynchronize();
  cudaFree(input_d);
  cudaFree(output_d);
  return T((*output_h));
}

// flash_attn_v2:
// feature ：增加thread在Br方向的tile, 减少online_softmax迭代次数，解决多次乘加带来的精度损失。同时减少对shared_memory的访问，性能提升150ms->38ms
template<typename T, const int Br = 16, const int Bc = 32, 
        const int TM = 1, const int TN = 2, const int KBD = 8, const int VBD = 8, const int paddingk = 0, const int paddingv = 0>
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
  float *s_v = s_k + (BN + paddingk) * KBD;
  float *s_o = s_v + (BN + paddingv) * VBD;
  float *s_m = s_o + BM * dim;
  float *s_l = s_m + BM;
  
  #pragma unroll
  for(size_t i = tid;i < BM;i += block_size) {
    s_m[i] = -__FLT_MAX__;
    s_l[i] = 0.f;
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
    
    /*
    #pragma unroll
    for(size_t i = tid;i < BN * dim;i += block_size) {
      int x = i % dim;
      int y = i / dim;
      s_k[x * (BN + 4) + y] = ((kv_acc_len + y) < kv_len) ? static_cast<float>(k[y * kv_stride + x]) : float(0);
      s_v[x * (BN + 4) + y] = ((kv_acc_len + y) < kv_len) ? static_cast<float>(v[y * kv_stride + x]) : float(0);
    }
    __syncthreads();
    */
    
    // sum[0]: ty tx 、 sum[1]: ty tx + Bc
    float sum[TM][TN] = {0.f};
    #pragma unroll
    for(size_t d = 0;d < dim;d += KBD) {
        #pragma unroll
        for(size_t i = tid;i < KBD * BN;i+=block_size) {
            int s_x = i % KBD;
            int y = i / KBD;
            s_k[s_x * (BN + paddingk) + y] = ((kv_acc_len + y && s_x + d < dim) < kv_len) ? static_cast<float>(k[y * kv_stride + s_x + d]) : float(0);
        }
        __syncthreads();
        #pragma unroll
        for(size_t i = 0;i < KBD;++i) {
            float q_reg[TM] = {0.f};
            float k_reg[TN] = {0.f};
            #pragma unroll
            for(size_t j = 0;j < TM;j++)
                q_reg[j] = (d + i < dim) ? s_q[(ty * TM + j) * dim + d + i] : 0.f;
            #pragma unroll
            for(size_t j = 0;j < TN;j++)
                k_reg[j] = s_k[i * (BN + paddingk) + tx * TN + j];
        
            #pragma unroll
            for(size_t j = 0;j < TM;j++) {
                #pragma unroll
                for(size_t k = 0;k < TN;k++) {
                    sum[j][k] += q_reg[j] * k_reg[k];
                }
            }
        }
        __syncthreads();
    }
    
    float l_pre[TM];//tile每行上一个Bc的和
    float l[TM];//加上当前Bc整体tile行的和
    float p_now[TM][TN];
    float exp_mprem[TM];
    float exp_mnowm[TM];
    #pragma unroll
    for(size_t i = 0;i < TM;i++) {
        float m_sum[TN]; //带casual掩码的sum
        float m_now;//tile每行的最大值
        m_now = -__FLT_MAX__;
        float l_now = 0.f;//tile每行的和
        float m_pre;//tile每行上一个Bc的最大值
        m_pre = s_m[ty * TM + i];
        l_pre[i] = s_l[ty * TM + i];
        float m;//加上当前Bc整体tile行的最大值
        m = m_pre;
        l[i] = l_pre[i];
        #pragma unroll
        for(size_t j = 0;j < TN;j++) {
            sum[i][j] *= scale;
            m_sum[j] = (((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * TN + j < kv_len)) && (!is_causal || q_acc_len + ty * TM + i >= kv_acc_len + tx * TN + j)) ? sum[i][j] : -__FLT_MAX__;
            m_now = fmaxf(m_now, m_sum[j]);
            sum[i][j] = (((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * TN + j < kv_len)) && (!is_causal || q_acc_len + ty * TM + i >= kv_acc_len + tx * TN + j)) ? expf(sum[i][j]) : 0.f;
            l_now += sum[i][j];
        }
        m_now = warp_reduce_max<float>(m_now);
        float expf_m_now = expf(-m_now);
        l_now *= expf_m_now;
        l_now = warp_reduce_sum<float>(l_now);
        #pragma unroll
        for(size_t j = 0;j < TN;j++)
            p_now[i][j] = sum[i][j] * expf_m_now;
        m = fmaxf(m, m_now);
        exp_mprem[i] = expf(m_pre - m);
        exp_mnowm[i] = expf(m_now - m);
        l[i] = l_pre[i] * exp_mprem[i] + l_now * exp_mnowm[i];
        s_m[ty * TM + i] = m;
        s_l[ty * TM + i] = l[i];
    }

    #pragma unroll
    for(size_t d = 0;d < dim;d += VBD) {
        #pragma unroll
        for(size_t i = tid;i < VBD * BN;i += block_size) {
            int s_x = i % VBD;
            int y = i / VBD;
            s_v[s_x * (BN + 1) + y] = ((kv_acc_len + y) < kv_len && s_x + d < dim) ? static_cast<float>(v[y * kv_stride + s_x + d]) : float(0);
        }
        __syncthreads();
        #pragma unroll
        for(size_t j = 0;j < VBD;++j) {
            float value[TM] = {0.f};
            #pragma unroll
            for(size_t i = 0;i < TN;i++) {
                float tmp = s_v[j * (BN + 1) + tx * TN + i];
                #pragma unroll
                for(size_t k = 0;k < TM;k++) {
                    value[k] += p_now[k][i] * tmp;
                }
            }
            #pragma unroll
            for(size_t i = 0;i < TM;i++) {
                value[i] = warp_reduce_sum<float>(value[i]);
                if(laneid == 0 && j + d < dim)
                    s_o[(ty * TM + i) * dim + j + d] = (q_acc_len + ty * TM + i < q_len) ? (s_o[(ty * TM + i) * dim + j + d] * exp_mprem[i] * l_pre[i] + value[i] * exp_mnowm[i]) / l[i] : 0.f;   
            }
        }
        __syncthreads();
    }
    k += BN * kv_stride;
    v += BN * kv_stride;
    kv_acc_len += BN;
    __syncthreads();
  }
  
  #pragma unroll
  for(size_t i = tid;i < BM * dim;i += block_size) {
    int x = i % dim;
    int y = i / dim;
    if(q_acc_len + y < q_len) {
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
  //if(is_causal) {
    //printf("is_causal\n");
    //return ;
  //}
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
  // (max_sram-8*2) / 2 : 6136
  // (max_sram-16*2) / 2 : 6128
  // (max_sram-32*2) / 2 : 6112
  // TM = (6128 / head_dim - 16) / 32
  // TM = (6112 / head_dim - 32) / 32
  switch (head_dim)
  {
  case 1:
    {
      constexpr int Br = 16;
      constexpr int Bc = 32;
      constexpr int TM = 4;
      constexpr int TN = 4;
      constexpr int KBD = 1;
      constexpr int VBD = 1;
      constexpr int paddingk = 0;
      constexpr int paddingv = 0;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 2:
    // TM = (6128 / 2 - 16) / 32 = 95
    {
      constexpr int Br = 16;
      constexpr int Bc = 32;
      constexpr int TM = 4;
      constexpr int TN = 4;
      constexpr int KBD = 2;
      constexpr int VBD = 2;
      constexpr int paddingk = 0;
      constexpr int paddingv = 0;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 4:
    {
      //12288 - qo: 4096 - kv: 4096 - lm: 1024
      constexpr int Br = 16;
      constexpr int Bc = 32;
      constexpr int TM = 4;
      constexpr int TN = 4;
      constexpr int KBD = 4;
      constexpr int VBD = 4;
      constexpr int paddingk = 0;
      constexpr int paddingv = 0;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 8:
    {
      //12288 - qo: 4096 - kv: 4096 - lm: 512  512 
      constexpr int Br = 16;
      constexpr int Bc = 32;
      constexpr int TM = 4;
      constexpr int TN = 4;
      constexpr int KBD = 8;
      constexpr int VBD = 8;
      constexpr int paddingk = 4;
      constexpr int paddingv = 4;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 16:
    {
      //12288 - qo: 8192 - kv: 2048 - lm: 512 = 1532
      constexpr int Br = 16;
      constexpr int Bc = 32;
      constexpr int TM = 4;
      constexpr int TN = 4;
      constexpr int KBD = 16;
      constexpr int VBD = 4;
      constexpr int paddingk = 4;
      constexpr int paddingv = 2;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 32:
    {
      //12288 - qo: 8192 - kv: 2048 - lm: 256 = 1792 4096 6144 128 32 8
      //2048 
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 16;
      constexpr int KBD = 8;
      constexpr int VBD = 8;
      constexpr int paddingk = 2;
      constexpr int paddingv = 1;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 64:
    {
      //12288 - qo: 2048 - kv: 8192 - lm: 128 = 1920 4096 6144 64 32 16
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 4;
      constexpr int KBD = 32;
      constexpr int VBD = 16;
      constexpr int paddingk = 1;
      constexpr int paddingv = 1;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN) * (KBD + VBD) + Br * TM * 2 + paddingk * KBD + paddingv * VBD) * sizeof(float);
      if(sram_bytes > max_sram_bytes) printf("sram over!\n");
      sram_bytes = min(sram_bytes, max_sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, KBD, VBD, paddingk, paddingv><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  default:
    printf("head_dim : %d\n", head_dim);
    break;
  }
    cudaMemcpy(h_o.data(), d_o, qo_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
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
