#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#define CEIL(N, M) (((N) + (M) - 1) / (M))
#define WARP_SZIE 32
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

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
    value += (x < n) ? input[x * cols + x] : 0.f;
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
    value = laneid < CEIL(blockDim.x, WARP_SZIE) ? smem[laneid] : T(0);
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
    grid.x = (CEIL(CEIL(diagonal, 8), 1024));
    block.x = 1024;
    const int NUM_PER_WARP = 8;
    const int STRIDE = 1024 * grid.x;
    trace_kernel<T><<<grid, block>>>(input_d, output_d, cols, diagonal, STRIDE, NUM_PER_WARP);
  }
  cudaMemcpy(output_h, output_d, sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(input_d);
  cudaFree(output_d);
  return T((*output_h));
}

// flash_attn_v2:
// 算法：grid<seqlen/Br, head_size, batch>, 每个block大小Br*Bc计算Br行的结果，block内求和/求最大值使用reduce规约计算
// feature ：
// 1. 每个thread处理TM*TN大小的tile, 减少online_softmax迭代次数，减少多次乘加带来的精度损失。同时减少对shared_memory的访问
// 2. 对k, v的shared_memory进行复用，只用一个s_kv，同时降低dim方向的维度(dim->8, 配合FLOAT4向量化访存可以实现对全局内存的合并访存),shared_memory的使用减少(dim/8)*2倍,用来增大thread tile
// 3. 对shared_memory进行padding, 达到bank conflict free
// 4. 利用feature2, 进行double buffer, 使访存和计算并行
template<typename T, const int Br = 16, const int Bc = 32, 
        const int TM = 2, const int TN = 8, const int BD = 8, const int padding = 4>
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
  
  const int Tc = CEIL((is_causal ? min((bx + 1) * BM, kv_len) : kv_len), BN);
  
  extern __shared__ char smem[];
  
  // shared_mem shape: for bcf 
  // q o: Br * dim
  // k v: dim * Bc
  float *s_q = (float*)smem;
  float *s_kv = s_q + BM * dim;
  float *s_o = s_kv + (BN + padding) * BD * 2;
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
    s_q[y * dim + x] = ((q_acc_len + y) < q_len) ? static_cast<float>(q[y * q_stride + x]) : float(0);
  }
  #pragma unroll
  for(size_t c = 0;c < Tc;++c) {
    float sum[TM][TN] = {0.f};
    #pragma unroll
    for(size_t d = 0;d < dim;d += BD) {
        int idx = (d / BD) % 2;
        #pragma unroll
        for(size_t i = tid;i < BD * BN;i+=block_size) {
            int s_x = i % BD;
            int y = i / BD;
            s_kv[idx * (BN + padding) * BD + s_x * (BN + padding) + y] = (kv_acc_len + y < kv_len && s_x + d < dim) ? static_cast<float>(k[y * kv_stride + s_x + d]) : float(0);
        }
        __syncthreads();
        #pragma unroll
        for(size_t i = 0;i < BD;++i) {
            float q_reg[TM] = {0.f};
            float k_reg[TN] = {0.f};
            #pragma unroll
            for(size_t j = 0;j < TM;j++)
                q_reg[j] = (d + i < dim) ? s_q[(ty * TM + j) * dim + d + i] : 0.f;
            #pragma unroll
            for(size_t j = 0;j < TN;j+=4)
                LDST128BITS(k_reg[j]) = LDST128BITS(s_kv[idx * (BN + padding) * BD + i * (BN + padding) + tx * 4 + j * Bc]);
            
            #pragma unroll
            for(size_t j = 0;j < TM;j++) {
                #pragma unroll
                for(size_t k = 0;k < TN;k++) {
                    sum[j][k] += q_reg[j] * k_reg[k];
                }
            }
        }
    }
    
    float l_pre[TM];//tile每行上一个Bc的和
    float l[TM];//加上当前Bc整体tile行的和
    float p_now[TM][TN];
    float exp_mprem[TM];
    float exp_mnowm[TM];
    #pragma unroll
    for(size_t i = 0;i < TM;i++) {
        float m_sum; //带casual掩码的sum
        float m_now = -__FLT_MAX__;//tile每行的最大值
        float l_now = 0.f;//tile每行的和
        float m_pre = s_m[ty * TM + i];//tile每行上一个Bc的最大值
        l_pre[i] = s_l[ty * TM + i];
        float m = m_pre;//加上当前Bc整体tile行的最大值
        l[i] = l_pre[i];
        #pragma unroll
        for(size_t j = 0;j < TN;++j) {
            sum[i][j] *= scale;
            m_sum = (((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4)) ? sum[i][j] : -__FLT_MAX__;
            if(((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4))
                m_now = fmaxf(m_now, m_sum);
        }
        m_now = warp_reduce_max<float>(m_now);
        for(size_t j = 0;j < TN;++j) {
            sum[i][j] = (((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4)) ? expf(sum[i][j]-m_now) : 0.f;
            if(((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4))
                l_now += sum[i][j];
        }
        l_now = warp_reduce_sum<float>(l_now);
        #pragma unroll
        for(size_t j = 0;j < TN;j++)
            p_now[i][j] = sum[i][j];
        m = fmaxf(m, m_now);
        exp_mprem[i] = expf(m_pre - m);
        exp_mnowm[i] = expf(m_now - m);
        l[i] = l_pre[i] * exp_mprem[i] + l_now * exp_mnowm[i];
        s_m[ty * TM + i] = m;
        s_l[ty * TM + i] = l[i];
    }
    
    #pragma unroll
    for(size_t d = 0;d < dim;d += BD) {
        int idx = (d / BD) % 2;
        #pragma unroll
        for(size_t i = tid;i < BD * BN;i += block_size) {
            int s_x = i % BD;
            int y = i / BD;
            s_kv[idx * (BN + padding) * BD + s_x * (BN + padding) + y] = ((kv_acc_len + y) < kv_len && s_x + d < dim) ? static_cast<float>(v[y * kv_stride + s_x + d]) : float(0);
        }
        __syncthreads();
        #pragma unroll
        for(size_t j = 0;j < BD;++j) {
            float value[TM] = {0.f};
            #pragma unroll
            for(size_t i = 0;i < TN;i+=4) {
                float4 tmp = LDST128BITS(s_kv[idx * (BN + padding) * BD + j * (BN + padding) + tx * 4 + i * Bc]);
                #pragma unroll
                for(size_t k = 0;k < TM;k++) {
                    value[k] += p_now[k][i] * tmp.x + p_now[k][i + 1] * tmp.y + p_now[k][i + 2] * tmp.z + p_now[k][i + 3] * tmp.w;
                }
            }
            #pragma unroll
            for(size_t i = 0;i < TM;i++) {
                value[i] = warp_reduce_sum<float>(value[i]);
                if(laneid == 0 && j + d < dim)
                    s_o[(ty * TM + i) * dim + j + d] = (q_acc_len + ty * TM + i < q_len) ? (s_o[(ty * TM + i) * dim + j + d] * exp_mprem[i] * l_pre[i] + value[i] * exp_mnowm[i]) / l[i] : 0.f;   
            }
        }
    }
    k += BN * kv_stride;
    v += BN * kv_stride;
    kv_acc_len += BN;
  }
  __syncthreads();
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
  
  switch (head_dim)
  {
  case 1:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 2;
      constexpr int TN = 8;
      constexpr int BD = 1;
      constexpr int padding = 0;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 2:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 2;
      constexpr int TN = 8;
      constexpr int BD = 2;
      constexpr int padding = 0;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 4:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 2;
      constexpr int TN = 8;
      constexpr int BD = 4;
      constexpr int padding = 0;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 8:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 2;
      constexpr int TN = 8;
      constexpr int BD = 8;
      constexpr int padding = 4;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 16:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 2;
      constexpr int TN = 8;
      constexpr int BD = 8;
      constexpr int padding = 4;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 32:
    { 
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 2;
      constexpr int TN = 16;
      constexpr int BD = 8;
      constexpr int padding = 4;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);//4096 + 128 + 
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);//24576
      cudaFuncSetAttribute(                                                      
        flash_attn_kernel<               
            T, Br, Bc, TM, TN, BD, padding>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
      int max_sram_size;
      cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
      printf("Max shared memory: %d, requested shared memory: %d \n",
           max_sram_size,
           sram_bytes);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
    }
    break;
  case 64:
    {
      constexpr int Br = 32;
      constexpr int Bc = 32;
      constexpr int TM = 1;
      constexpr int TN = 8;
      constexpr int BD = 8;
      constexpr int padding = 4;
      dim3 block(Br * Bc);
      dim3 grid(CEIL(target_seq_len, Br * TM), query_heads, batch_size);
      int sram_bytes = ((Br * TM) * head_dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
      flash_attn_kernel<T, Br, Bc, TM, TN, BD, padding><<<grid, block, sram_bytes>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, kv_heads, head_dim, is_causal, rsqrtf(head_dim));
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
