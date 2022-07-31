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



__global__ void intensity_transform_batch_forward_kernel(
		float *const output_ptr, 
		const float *const image_ptr,
		const float *const curve_ptr,
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
		for (int32_t bs = 0; bs < N; ++bs) {
			const int32_t bs_offset = bs * C * area_size;
			for (int32_t ch = 0; ch < C; ++ch) {
				const int32_t offset           = bs_offset + ch * area_size + idx_y * W + idx_x;
				const float *const curve_start = curve_ptr + (bs * C + ch) * 256;
				/* 加权 */
				down_index_ptr[offset]         = fmaxf(0, floor(image_ptr[offset]));
				up_index_ptr[offset]           = fminf(255, down_index_ptr[offset] + 1);
				up_weight_ptr[offset]          = image_ptr[offset] - down_index_ptr[offset];
				down_weight_ptr[offset]        = 1.f - up_weight_ptr[offset];
				output_ptr[offset]             = down_weight_ptr[offset] * curve_start[down_index_ptr[offset]]
							                   + up_weight_ptr[offset]   * curve_start[up_index_ptr[offset]];
			}
		}
	}
}




__global__ void intensity_transform_batch_forward_kernel_2(
		float *const output_ptr, 
		const float *const image_ptr,
		const float *const curve_ptr,
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
		const int batch_id   		= idx / batch_length;
		const int channel_id 		= (idx % batch_length) / area_length;
		/* 找到这个 batch 的第 channel 张的曲线 */
		const float *curve_start    = curve_ptr + (batch_id * C + channel_id) * 256;
		/* 加权 */
		down_index_ptr[idx]         = fmaxf(0, floor(image_ptr[idx]));
		up_index_ptr[idx]           = fminf(255, down_index_ptr[idx] + 1);
		up_weight_ptr[idx]          = image_ptr[idx] - down_index_ptr[idx];
		down_weight_ptr[idx]        = 1.f - up_weight_ptr[idx];
		output_ptr[idx]             = down_weight_ptr[idx] * curve_start[down_index_ptr[idx]]
					                + up_weight_ptr[idx]   * curve_start[up_index_ptr[idx]];
	}
}



void intensity_transform_batch_forward_cuda(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve,
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight,
		at::Tensor& up_weight) {
	assert(image.is_contiguous());
	assert(curve.is_contiguous());
	/* 获取图片维度 */	
	const int32_t N = image.size(0);
	const int32_t C = image.size(1);
	const int32_t H = image.size(2);
	const int32_t W = image.size(3);
	/* 决定 GPU 内存布局 */
	constexpr int32_t block_x = 16;
	constexpr int32_t block_y = 16;
	

	// /* 方案一 */
	// dim3 block(block_x, block_y);
	// dim3 grid(
	// 	int((W + block_x - 1) / block_x), 
	// 	int((H + block_y - 1) / block_y)
	// );
	// intensity_transform_batch_forward_kernel<<<grid, block>>>(
	// 	output.data_ptr<float>(),
	// 	image.data_ptr<float>(),
	// 	curve.data_ptr<float>(),
	// 	down_index.data_ptr<int32_t>(),
	// 	up_index.data_ptr<int32_t>(),
	// 	down_weight.data_ptr<float>(),
	// 	up_weight.data_ptr<float>(),
	// 	N, C, H, W
	// );

	/* 方案二, 更快 */
	dim3 block(block_x * block_y);
	dim3 grid(
		int((H * W+ block_x * block_y - 1) / (block_x * block_y)), /* 14 x 14 */
		N * C 													   /* 16 x 3 */
	);
	intensity_transform_batch_forward_kernel_2<<<grid, block>>>(
		output.data_ptr<float>(),
		image.data_ptr<float>(),
		curve.data_ptr<float>(),
		down_index.data_ptr<int32_t>(),
		up_index.data_ptr<int32_t>(),
		down_weight.data_ptr<float>(),
		up_weight.data_ptr<float>(),
		N * C * H * W, 
		C * H * W,
		H * W,
		C,
		grid.x * block_x * block_y
	);
}