// C & C++
#include <math.h>
#include <assert.h>
#include <cstdio>
#include <vector>
#include <iostream>
// CUDA
#include <cuda_runtime.h>
// Torch
#include <cuda.h>
#include <ATen/ATen.h>
#include <torch/torch.h>




__global__ void intensity_transform_batch_backward_kernel(
		float *const curve_grad_ptr, 
		const float *const grad_from_output_ptr,
		int32_t *const down_index_ptr, 
		int32_t *const up_index_ptr, 
		float *const down_weight_ptr,
		float *const up_weight_ptr,
		const int32_t N,
		const int32_t C, 
		const int32_t H,
		const int32_t W) {
	/* 计算当前线程的逻辑下标 */
	const int32_t idx_y = threadIdx.y + blockIdx.y * blockDim.y;
	const int32_t idx_x = threadIdx.x + blockIdx.x * blockDim.x;
	/* 下标在数据范围之内 */
	if (idx_y < H && idx_x < W) {
		/* 多个 batch 多个通道同时处理这一个位置的像素 */
		const int32_t area_size = H * W;
		for (int32_t b_idx = 0; b_idx < N; ++b_idx) {
			const int32_t b_offset = b_idx * C * area_size;
			for (int32_t ch = 0; ch < C; ++ch) {
				const int32_t offset  = b_offset + ch * area_size + idx_y * W + idx_x;
				float *curve_start    = curve_grad_ptr + (b_idx * C + ch) * 256;
				atomicAdd(curve_start + down_index_ptr[offset], grad_from_output_ptr[offset] * down_weight_ptr[offset]);
				atomicAdd(curve_start + up_index_ptr[offset],   grad_from_output_ptr[offset] * up_weight_ptr[offset]);
			}
		}
	}
}





__global__ void intensity_transform_batch_backward_kernel_2(
		float *const curve_grad_ptr, 
		const float *const grad_from_output_ptr,
		int32_t *const down_index_ptr, 
		int32_t *const up_index_ptr, 
		float *const down_weight_ptr,
		float *const up_weight_ptr,
		const int32_t length,
		const int32_t batch_length, 
		const int32_t area_length,
		const int32_t C, 
		const int32_t element_count_in_one_line) {
	/* 计算当前线程的逻辑下标 */
	const int32_t idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * element_count_in_one_line;
	if (idx < length) {
		/* 根据 idx 算出图像中这个数据所在的 batch 和 channel 编号 */
		const int batch_id    = idx / batch_length;
		const int channel_id  = (idx % batch_length) / area_length;
		/* 找到这个 batch 的第 channel 张的曲线 */
		float *curve_start    = curve_grad_ptr + (batch_id * C + channel_id) * 256;
		/* 加权 */
		atomicAdd(curve_start + down_index_ptr[idx], grad_from_output_ptr[idx] * down_weight_ptr[idx]);
		atomicAdd(curve_start + up_index_ptr[idx],   grad_from_output_ptr[idx] * up_weight_ptr[idx]);
	}
}





void intensity_transform_batch_backward_cuda(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		const at::Tensor& down_index, 
		const at::Tensor& up_index, 
		const at::Tensor& down_weight,
		const at::Tensor& up_weight) {
	/* 获取图片维度 */	
	const int32_t N = grad_from_output.size(0);
	const int32_t C = grad_from_output.size(1);
	const int32_t H = grad_from_output.size(2);
	const int32_t W = grad_from_output.size(3);
	/* 决定 GPU 内存布局 */
	constexpr int32_t block_x = 16;
	constexpr int32_t block_y = 16;
	
	/* 方案一 */
	dim3 block(block_x, block_y);
	dim3 grid(
		int((W + block_x - 1) / block_x), 
		int((H + block_y - 1) / block_y)
	);
	intensity_transform_batch_backward_kernel<<<grid, block>>>(
		curve_grad.data_ptr<float>(),
		grad_from_output.data_ptr<float>(),
		down_index.data_ptr<int32_t>(),
		up_index.data_ptr<int32_t>(),
		down_weight.data_ptr<float>(),
		up_weight.data_ptr<float>(),
		N, C, H, W
	);
}

