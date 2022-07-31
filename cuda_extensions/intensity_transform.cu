// C++
#include <math.h>
#include <assert.h>
#include <cstdio>
#include <iostream>
// CUDA
#include <cuda_runtime.h>
// Torch
#include <cuda.h>
#include <ATen/ATen.h>
#include <torch/torch.h>



__global__ void intensity_transform_forward_kernel(
		float *const output_ptr, 
		const float *const image_ptr, 
		const float *const curve_ptr, 
		int32_t *const down_index_ptr,
		int32_t *const up_index_ptr,
		float *const down_weight_ptr,
		float *const up_weight_ptr,
		const int32_t length) {
	/* 计算当前线程对应输入的索引 */
	int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	/* 没有超出输入范围的线程才做计算 */
	if (idx < length) {
		/* 根据 image 的值, 找到每一个像素值的上下界 */
		down_index_ptr[idx]  = fmaxf(0, floor(image_ptr[idx]));
		up_index_ptr[idx]        = fminf(255, down_index_ptr[idx] + 1);
		/* 计算上下界的加权参数 */
		up_weight_ptr[idx]   = image_ptr[idx] - down_index_ptr[idx];
		down_weight_ptr[idx] = 1.f - up_weight_ptr[idx];
		/* 加权映射 */
		output_ptr[idx]      = down_weight_ptr[idx] * curve_ptr[down_index_ptr[idx]]
					         + up_weight_ptr[idx]   * curve_ptr[up_index_ptr[idx]];
	}
}


void intensity_transform_forward_cuda(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length, 
		const int32_t block_size) {
	/* 决定 GPU 内存结构 */
	dim3 block(block_size);
	dim3 grid((length + block_size - 1) / block.x);

	/* 检查数据合法性, 是否连续、在不在同一个 cuda 设备上、支不支持 cuda 等 */

	/* 启动核函数 */
	intensity_transform_forward_kernel<<<grid, block>>>(
		output.data_ptr<float>(),
		image.data_ptr<float>(),
		curve.data_ptr<float>(),
		down_index.data_ptr<int32_t>(),
		up_index.data_ptr<int32_t>(),
		down_weight.data_ptr<float>(),
		up_weight.data_ptr<float>(),
		length
	);
}




__global__ void intensity_transform_backward_kernel(
		float *const curve_grad_ptr,
		const float *const grad_from_output_ptr,
		const int32_t *const down_index_ptr,
		const int32_t *const up_index_ptr,
		const float *const down_weight_ptr,
		const float *const up_weight_ptr,
		const int32_t length) {
	/* 计算当前线程对应输入的索引 */
	int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	/* 没有超出输入范围的线程才做计算 */
	if (idx < length) {
		atomicAdd(curve_grad_ptr + down_index_ptr[idx], grad_from_output_ptr[idx] * down_weight_ptr[idx]);
		atomicAdd(curve_grad_ptr + up_index_ptr[idx], grad_from_output_ptr[idx] * up_weight_ptr[idx]);
	}
}





void intensity_transform_backward_cuda(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length, 
		const int32_t block_size) {
	/* 决定 GPU 逻辑结构 */
	dim3 block(block_size);
	dim3 grid((length + block.x - 1) / block.x);

	/* 检查 */
	intensity_transform_backward_kernel<<<grid, block>>>(
		curve_grad.data_ptr<float>(), 
		grad_from_output.data_ptr<float>(),
		down_index.data_ptr<int32_t>(),
		up_index.data_ptr<int32_t>(),
		down_weight.data_ptr<float>(),
		up_weight.data_ptr<float>(),
		length
	);
}