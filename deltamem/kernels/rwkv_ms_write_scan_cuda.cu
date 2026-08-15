#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ int64_t feature_offset(
    int64_t batch,
    int64_t token,
    int64_t head,
    int64_t rank,
    int64_t seq_len,
    int64_t num_heads) {
  return (((batch * seq_len + token) * num_heads + head) * rank);
}

__device__ __forceinline__ int64_t state_offset(
    int64_t batch,
    int64_t head,
    int64_t slot,
    int64_t row,
    int64_t col,
    int64_t num_heads,
    int64_t num_slots,
    int64_t rank) {
  return ((((batch * num_heads + head) * num_slots + slot) * rank + row) * rank + col);
}

__device__ __forceinline__ int64_t history_offset(
    int64_t batch,
    int64_t token,
    int64_t head,
    int64_t row,
    int64_t col,
    int64_t seq_len,
    int64_t num_heads,
    int64_t rank) {
  return (((((batch * seq_len + token) * num_heads + head) * rank + row) * rank) + col);
}

__global__ void rwkv_write_forward_kernel(
    float* state,
    const float* w,
    const float* k,
    const float* v,
    const float* a,
    const float* b,
    const float* keep,
    const float* erase,
    const float* write,
    const int64_t* slots,
    float* history,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t num_slots,
    int64_t rank,
    float erase_gate,
    bool save_history) {
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t row = block % rank;
  block /= rank;
  int64_t slot = block % num_slots;
  block /= num_slots;
  int64_t head = block % num_heads;
  int64_t batch = block / num_heads;
  if (batch >= batch_size) {
    return;
  }

  for (int64_t token = 0; token < seq_len; ++token) {
    if (slots[batch * seq_len + token] != slot) {
      continue;
    }
    int64_t f = feature_offset(batch, token, head, rank, seq_len, num_heads);
    int64_t s = state_offset(batch, head, slot, row, 0, num_heads, num_slots, rank);
    int64_t h = history_offset(batch, token, head, row, 0, seq_len, num_heads, rank);
    float x = threadIdx.x < rank ? state[s + threadIdx.x] : 0.0f;
    if (save_history && threadIdx.x < rank) {
      history[h + threadIdx.x] = x;
    }
    float a_value = threadIdx.x < rank ? a[f + threadIdx.x] : 0.0f;
    float correction = warp_sum(x * a_value);
    if (threadIdx.x < rank) {
      float w_value = w[f + threadIdx.x];
      float k_value = k[f + threadIdx.x];
      float b_value = b[f + threadIdx.x];
      float row_keep = keep[f + row];
      float row_erase = erase[f + row];
      float row_write = write[f + row];
      float row_v = v[f + row];
      float state_term = __fmul_rn(__fmul_rn(row_keep, w_value), x);
      float write_outer = __fmul_rn(row_v, k_value);
      float write_term = __fmul_rn(row_write, write_outer);
      float correction_outer = __fmul_rn(correction, b_value);
      float erase_term = __fmul_rn(
          __fmul_rn(erase_gate, row_erase), correction_outer);
      state[s + threadIdx.x] = __fadd_rn(
          __fadd_rn(state_term, write_term), erase_term);
    }
  }
}

__global__ void rwkv_write_backward_kernel(
    const float* grad_state_out,
    float* grad_state0,
    const float* history,
    const float* w,
    const float* k,
    const float* v,
    const float* a,
    const float* b,
    const float* keep,
    const float* erase,
    const float* write,
    const int64_t* slots,
    float* grad_w,
    float* grad_k,
    float* grad_v,
    float* grad_a,
    float* grad_b,
    float* grad_keep,
    float* grad_erase,
    float* grad_write,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t num_slots,
    int64_t rank,
    float erase_gate) {
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t row = block % rank;
  block /= rank;
  int64_t slot = block % num_slots;
  block /= num_slots;
  int64_t head = block % num_heads;
  int64_t batch = block / num_heads;
  if (batch >= batch_size) {
    return;
  }

  int64_t final_s = state_offset(batch, head, slot, row, 0, num_heads, num_slots, rank);
  int64_t final_g = final_s;
  float grad = threadIdx.x < rank ? grad_state_out[final_g + threadIdx.x] : 0.0f;
  for (int64_t token = seq_len - 1; token >= 0; --token) {
    if (slots[batch * seq_len + token] != slot) {
      continue;
    }
    int64_t f = feature_offset(batch, token, head, rank, seq_len, num_heads);
    int64_t h = history_offset(batch, token, head, row, 0, seq_len, num_heads, rank);
    float x = threadIdx.x < rank ? history[h + threadIdx.x] : 0.0f;
    float w_value = threadIdx.x < rank ? w[f + threadIdx.x] : 0.0f;
    float k_value = threadIdx.x < rank ? k[f + threadIdx.x] : 0.0f;
    float a_value = threadIdx.x < rank ? a[f + threadIdx.x] : 0.0f;
    float b_value = threadIdx.x < rank ? b[f + threadIdx.x] : 0.0f;
    float correction = warp_sum(x * a_value);
    float grad_b_dot = warp_sum(grad * b_value);
    float row_keep = keep[f + row];
    float row_erase = erase[f + row];
    float row_write = write[f + row];
    float row_v = v[f + row];
    float grad_keep_value = warp_sum(grad * w_value * x);
    float grad_write_value = warp_sum(grad * row_v * k_value);
    float grad_v_value = warp_sum(grad * row_write * k_value);
    if (threadIdx.x == 0) {
      atomicAdd(grad_keep + f + row, grad_keep_value);
      atomicAdd(grad_write + f + row, grad_write_value);
      atomicAdd(grad_v + f + row, grad_v_value);
      atomicAdd(grad_erase + f + row, erase_gate * correction * grad_b_dot);
    }
    if (threadIdx.x < rank) {
      atomicAdd(grad_w + f + threadIdx.x, grad * row_keep * x);
      atomicAdd(grad_k + f + threadIdx.x, grad * row_write * row_v);
      atomicAdd(grad_a + f + threadIdx.x, erase_gate * row_erase * grad_b_dot * x);
      atomicAdd(grad_b + f + threadIdx.x, erase_gate * row_erase * correction * grad);
      grad =
          grad * row_keep * w_value
          + erase_gate * row_erase * grad_b_dot * a_value;
    }
  }
  if (threadIdx.x < rank) {
    grad_state0[final_g + threadIdx.x] = grad;
  }
}

std::vector<torch::Tensor> forward(
    torch::Tensor state,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor keep,
    torch::Tensor erase,
    torch::Tensor write,
    torch::Tensor slots,
    double erase_gate,
    bool save_history) {
  TORCH_CHECK(state.is_cuda(), "RWKV write scan requires CUDA state");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "RWKV write scan requires float32 state");
  TORCH_CHECK(state.dim() == 5, "state must have shape [batch, heads, slots, rank, rank]");
  TORCH_CHECK(slots.scalar_type() == torch::kInt64, "slots must be int64");
  auto state_out = state.contiguous().clone();
  const auto batch_size = state.size(0);
  const auto num_heads = state.size(1);
  const auto num_slots = state.size(2);
  const auto rank = state.size(3);
  const auto seq_len = w.size(1);
  TORCH_CHECK(rank == state.size(4), "state must be square");
  TORCH_CHECK(rank <= 32, "RWKV write scan currently supports rank <= 32");
  const std::vector<int64_t> expected_features = {
      batch_size, seq_len, num_heads, rank};
  for (const auto& feature : {w, k, v, a, b, keep, erase, write}) {
    TORCH_CHECK(feature.is_cuda(), "RWKV write scan features must be CUDA tensors");
    TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "RWKV write scan features must be float32");
    TORCH_CHECK(feature.sizes() == expected_features, "RWKV write scan feature shape differs");
    TORCH_CHECK(feature.device() == state.device(), "RWKV write scan tensors must share a device");
  }
  TORCH_CHECK(slots.is_cuda() && slots.device() == state.device(), "slots must share the CUDA device");
  TORCH_CHECK(slots.sizes() == torch::IntArrayRef({batch_size, seq_len}), "slot shape differs");
  auto history = save_history
      ? torch::empty({batch_size, seq_len, num_heads, rank, rank}, state.options())
      : torch::empty({0}, state.options());
  const dim3 grid(batch_size * num_heads * num_slots * rank);
  const dim3 block(32);
  auto stream = at::cuda::getCurrentCUDAStream();
  rwkv_write_forward_kernel<<<grid, block, 0, stream>>>(
      state_out.data_ptr<float>(), w.data_ptr<float>(), k.data_ptr<float>(),
      v.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
      keep.data_ptr<float>(), erase.data_ptr<float>(), write.data_ptr<float>(),
      slots.data_ptr<int64_t>(), history.data_ptr<float>(), batch_size, seq_len,
      num_heads, num_slots, rank, static_cast<float>(erase_gate), save_history);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {state_out, history};
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_state,
    torch::Tensor history,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor keep,
    torch::Tensor erase,
    torch::Tensor write,
    torch::Tensor slots,
    double erase_gate) {
  TORCH_CHECK(grad_state.is_cuda() && grad_state.scalar_type() == torch::kFloat32,
              "RWKV write scan output gradient must be CUDA float32");
  TORCH_CHECK(history.is_cuda() && history.scalar_type() == torch::kFloat32,
              "RWKV write scan history must be CUDA float32");
  auto grad_state0 = grad_state.contiguous().clone();
  auto grad_w = torch::zeros_like(w);
  auto grad_k = torch::zeros_like(k);
  auto grad_v = torch::zeros_like(v);
  auto grad_a = torch::zeros_like(a);
  auto grad_b = torch::zeros_like(b);
  auto grad_keep = torch::zeros_like(keep);
  auto grad_erase = torch::zeros_like(erase);
  auto grad_write = torch::zeros_like(write);
  const auto batch_size = grad_state.size(0);
  const auto num_heads = grad_state.size(1);
  const auto num_slots = grad_state.size(2);
  const auto rank = grad_state.size(3);
  const auto seq_len = w.size(1);
  const dim3 grid(batch_size * num_heads * num_slots * rank);
  const dim3 block(32);
  auto stream = at::cuda::getCurrentCUDAStream();
  rwkv_write_backward_kernel<<<grid, block, 0, stream>>>(
      grad_state.data_ptr<float>(), grad_state0.data_ptr<float>(),
      history.data_ptr<float>(), w.data_ptr<float>(), k.data_ptr<float>(),
      v.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
      keep.data_ptr<float>(), erase.data_ptr<float>(), write.data_ptr<float>(),
      slots.data_ptr<int64_t>(), grad_w.data_ptr<float>(), grad_k.data_ptr<float>(),
      grad_v.data_ptr<float>(), grad_a.data_ptr<float>(), grad_b.data_ptr<float>(),
      grad_keep.data_ptr<float>(), grad_erase.data_ptr<float>(), grad_write.data_ptr<float>(),
      batch_size, seq_len, num_heads, num_slots, rank, static_cast<float>(erase_gate));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_state0, grad_w, grad_k, grad_v, grad_a, grad_b,
          grad_keep, grad_erase, grad_write};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "RWKV multi-state write scan forward");
  m.def("backward", &backward, "RWKV multi-state write scan backward");
}
