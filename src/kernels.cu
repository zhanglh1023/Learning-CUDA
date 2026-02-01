#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#define CEIL(N, M) (((N) + (M) - 1) / (M))
#define WARP_SZIE 32

template<typename T>
__device__ __forceinline__ T warp_reduce(T value) {
  #pragma unroll
  for(size_t i = WARP_SZIE >> 1;i > 0;i >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, i);
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
    if(x < n) value += input[x];
  }
  value = warp_reduce<T>(value);
  const int warpid = tid / 32;
  const int laneid = tid % 32;
  __shared__ T smem[32];
  if(laneid == 0) {
    smem[warpid] = value;
  }
  __syncthreads();
  if(warpid == 0) {
    value = smem[laneid];
    value = warp_reduce<T>(value);
    if(laneid == 0) *output = value;
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
  size_t diagonal = min(rows, cols);
  printf("diagonal: %d\n", diagonal);
  dim3 block(1024);
  dim3 grid(1);
  if(diagonal <= 4096) {
    const int STRIDE = 1024;
    const int NUM_PER_WARP = CEIL(diagonal, 1024);
    trace_kernel<T><<<grid, block>>>(input_d, output_d, cols, diagonal, STRIDE, NUM_PER_WARP);
  } else {
    grid.x = (CEIL(CEIL(diagonal, 4), 1024));
    const int NUM_PER_WARP = 4;
    const int STRIDE = 1024 * grid.x;
    trace_kernel<T><<<grid, block>>>(input_d, output_d, cols, diagonal, STRIDE, NUM_PER_WARP);
  }
  T *output_h = (T*)malloc(sizeof(T));
  cudaMemcpy(output_h, output_d, sizeof(T), cudaMemcpyDeviceToHost);
  return T((*output_h));
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
  // TODO: Implement the flash attention function
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
