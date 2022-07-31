// C++
#include <cmath>
#include <vector>
#include <iostream>
// torch
#include <torch/extension.h>


void intensity_transform_forward_cpu(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length) {
	/* 获取张量的指针 */
	float       *const output_ptr      = output.data_ptr<float>();
	const float *const image_ptr  	   = image.data_ptr<float>();
	const float *const curve_ptr       = curve.data_ptr<float>();
	int32_t 	*const down_index_ptr  = down_index.data_ptr<int32_t>();
	int32_t 	*const up_index_ptr    = up_index.data_ptr<int32_t>();
	float 		*const down_weight_ptr = down_weight.data_ptr<float>();
	float 		*const up_weight_ptr   = up_weight.data_ptr<float>();
	/* 根据 image 的值, 找到每一个像素值的上下界 */
	for (int32_t i = 0; i < length; ++i) {
		down_index_ptr[i] = std::fmax(0, std::floor(image_ptr[i]));
		up_index[i]       = std::fmin(255, down_index_ptr[i] + 1);
	}
	/* 计算上下界的加权参数 */
	for (int32_t i = 0; i < length; ++i) {
		up_weight_ptr[i]   = image_ptr[i] - down_index_ptr[i];
		down_weight_ptr[i] = 1.f - up_weight_ptr[i];
	}
	/* 加权映射 */
	for (int32_t i = 0; i < length; ++i) {
		output_ptr[i] = down_weight_ptr[i] * curve_ptr[down_index_ptr[i]]
					  + up_weight_ptr[i]   * curve_ptr[up_index_ptr[i]];
	}
}


void intensity_transform_backward_cpu(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length) {
	/* 获取张量的指针 */
	float       *const curve_grad_ptr       = curve_grad.data_ptr<float>();
	const float *const grad_from_output_ptr = grad_from_output.data_ptr<float>();
	int32_t 	*const down_index_ptr  		= down_index.data_ptr<int32_t>();
	int32_t 	*const up_index_ptr    		= up_index.data_ptr<int32_t>();
	float 		*const down_weight_ptr 		= down_weight.data_ptr<float>();
	float 		*const up_weight_ptr   		= up_weight.data_ptr<float>();
	/* 找上下界, 根据权值乘以对应的梯度, 得到曲线中每个上下界(节点)的梯度 */
	for (int32_t i = 0; i < length; ++i) {
		const int32_t index    = down_index_ptr[i];
		curve_grad_ptr[index] += grad_from_output_ptr[i] * down_weight_ptr[i];
	}
	for (int32_t i = 0; i < length; ++i) {
		const int32_t index    = up_index_ptr[i];
		curve_grad_ptr[index] += grad_from_output_ptr[i] * up_weight_ptr[i];
	}
}



void intensity_transform_batch_forward_cpu(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve,
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight,
		at::Tensor& up_weight) {
	/* 获取图像信息 */
	const int32_t N 				= image.size(0);
	const int32_t C 				= image.size(1);
	const int32_t area_size 		= image.size(2) * image.size(3);
	const int32_t batch_len 		= C * area_size;
	constexpr int32_t one_curve_len = 256;
	/* 暴力逐 batch 逐通道做映射 */
	for (int32_t b_idx = 0; b_idx < N; ++b_idx) {
		const int32_t b_offset = b_idx * batch_len;
		for (int32_t ch = 0; ch < C; ++ch) {
			const int32_t offset = b_offset + ch * area_size;
			float 		*const output_ptr      = output.data_ptr<float>() + offset;
			const float *const image_ptr       = image.data_ptr<float>() + offset;
			const float *const curve_ptr  	   = curve.data_ptr<float>() + (b_idx * C + ch) * one_curve_len;
			int32_t     *const down_index_ptr  = down_index.data_ptr<int32_t>() + offset;
			int32_t     *const up_index_ptr    = up_index.data_ptr<int32_t>() + offset;
			float 		*const down_weight_ptr = down_weight.data_ptr<float>() + offset;
			float 		*const up_weight_ptr   = up_weight.data_ptr<float>() + offset;
			for (int32_t i = 0; i < area_size; ++i) {
				/* 根据 image 的值, 找到每一个像素值的上下界 */
				down_index_ptr[i]  = fmaxf(0, floor(image_ptr[i]));
				up_index_ptr[i]    = fminf(255, down_index_ptr[i] + 1);
				/* 计算上下界的加权参数 */
				up_weight_ptr[i]   = image_ptr[i] - down_index_ptr[i];
				down_weight_ptr[i] = 1.f - up_weight_ptr[i];
				/* 加权映射 */
				output_ptr[i]      = down_weight_ptr[i] * curve_ptr[down_index_ptr[i]]
							       + up_weight_ptr[i]   * curve_ptr[up_index_ptr[i]];
			}
		}
	}
}


void intensity_transform_batch_backward_cpu(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		const at::Tensor& down_index, 
		const at::Tensor& up_index, 
		const at::Tensor& down_weight,
		const at::Tensor& up_weight) {
	/* 获取图像信息 */
	const int32_t N 				= grad_from_output.size(0);
	const int32_t C 				= grad_from_output.size(1);
	const int32_t area_size 		= grad_from_output.size(2) * grad_from_output.size(3);
	const int32_t batch_len 		= C * area_size;
	constexpr int32_t one_curve_len = 256;
	/* 暴力逐 batch 逐通道做映射 */
	for (int32_t b_idx = 0; b_idx < N; ++b_idx) {
		const int32_t b_offset = b_idx * batch_len;
		for (int32_t ch = 0; ch < C; ++ch) {
			const int32_t offset = b_offset + ch * area_size;
			const float   *const grad_from_output_ptr = grad_from_output.data_ptr<float>() + offset;
			float         *const curve_grad_ptr       = curve_grad.data_ptr<float>() + (b_idx * C + ch) * one_curve_len;
			const int32_t *const down_index_ptr  	  = down_index.data_ptr<int32_t>() + offset;
			const int32_t *const up_index_ptr    	  = up_index.data_ptr<int32_t>() + offset;
			const float   *const down_weight_ptr 	  = down_weight.data_ptr<float>() + offset;
			const float   *const up_weight_ptr   	  = up_weight.data_ptr<float>() + offset;
			
			for (int32_t i = 0; i < area_size; ++i) {
				const int32_t index    = down_index_ptr[i];
				curve_grad_ptr[index] += grad_from_output_ptr[i] * down_weight_ptr[i];
			}
			for (int32_t i = 0; i < area_size; ++i) {
				const int32_t index    = up_index_ptr[i];
				curve_grad_ptr[index] += grad_from_output_ptr[i] * up_weight_ptr[i];
			}
		}
	}
}




















#ifdef WITH_CUDA
void intensity_transform_forward_cuda(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length, 
		const int32_t block_size);

void intensity_transform_backward_cuda(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length, 
		const int32_t block_size);

void intensity_transform_batch_forward_cuda(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve,
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight,
		at::Tensor& up_weight);

void intensity_transform_batch_backward_cuda(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		const at::Tensor& down_index, 
		const at::Tensor& up_index, 
		const at::Tensor& down_weight,
		const at::Tensor& up_weight);
#endif












void intensity_transform_forward(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length, 
		const int32_t block_size) {
	if (image.is_cuda()) {
#ifdef WITH_CUDA
		intensity_transform_forward_cuda(output, image, curve, down_index, up_index, down_weight, up_weight, length, block_size);
#else
		AT_ERROR("Function 'intensity_transform_forward' is not complied with GPU support!")
#endif
	}
	else {
		intensity_transform_forward_cpu(output, image, curve, down_index, up_index, down_weight, up_weight, length);
	}
}



void intensity_transform_backward(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight, 
		const int32_t length,
		const int32_t block_size) {
	if (grad_from_output.is_cuda()) {
#ifdef WITH_CUDA
		intensity_transform_backward_cuda(curve_grad, grad_from_output, down_index, up_index, down_weight, up_weight, length, block_size);
#else
		AT_ERROR("Function 'intensity_transform_backward' is not complied with GPU support!")
#endif
	}
	else {
		intensity_transform_backward_cpu(curve_grad, grad_from_output, down_index, up_index, down_weight, up_weight, length);
	}
}





void intensity_transform_batch_forward(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& curve,
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight,
		at::Tensor& up_weight) {
	if (image.is_cuda()) {
#ifdef WITH_CUDA
		intensity_transform_batch_forward_cuda(output, image, curve, down_index, up_index, down_weight, up_weight);
#else
		AT_ERROR("Function 'intensity_transform_batch_forward' is not complied with GPU support!")
#endif
	}
	else {
		intensity_transform_batch_forward_cpu(output, image, curve, down_index, up_index, down_weight, up_weight);
	}
}



void intensity_transform_batch_backward(
		at::Tensor& curve_grad, 
		const at::Tensor& grad_from_output, 
		at::Tensor& down_index, 
		at::Tensor& up_index, 
		at::Tensor& down_weight, 
		at::Tensor& up_weight) {
	if (grad_from_output.is_cuda()) {
#ifdef WITH_CUDA
		intensity_transform_batch_backward_cuda(curve_grad, grad_from_output, down_index, up_index, down_weight, up_weight);
#else
		AT_ERROR("Function 'intensity_transform_backward' is not complied with GPU support!")
#endif
	}
	else {
		intensity_transform_batch_backward_cpu(curve_grad, grad_from_output, down_index, up_index, down_weight, up_weight);
	}
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"forward", 
		&intensity_transform_forward, 
		"forward for intensity transform", 
		py::arg("output"), 
		py::arg("image"), 
		py::arg("curve"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight"),  
		py::arg("length"),
		py::arg("block_size")=128
	);
	m.def(
		"backward", 
		&intensity_transform_backward, 
		"backward for intensity transform", 
		py::arg("curve_grad"), 
		py::arg("grad_from_output"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight"),  
		py::arg("length"),
		py::arg("block_size")=128
	);
	m.def(
		"batch_forward", 
		&intensity_transform_batch_forward, 
		"batch forward for intensity transform", 
		py::arg("output"), 
		py::arg("image"), 
		py::arg("curve"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight")
	);
	m.def(
		"batch_backward", 
		&intensity_transform_batch_backward, 
		"batch backward for intensity transform", 
		py::arg("curve_grad"), 
		py::arg("grad_from_output"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight")
	);
	m.def(
		"forward_cpu", 
		&intensity_transform_forward_cpu, 
		"forward for intensity transform with cpu support", 
		py::arg("output"), 
		py::arg("image"), 
		py::arg("curve"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight"),  
		py::arg("length")
	);
	m.def(
		"backward_cpu", 
		&intensity_transform_backward_cpu, 
		"backward for intensity transform with cpu support", 
		py::arg("curve_grad"), 
		py::arg("grad_from_output"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight"),  
		py::arg("length")
	);
	m.def(
		"batch_forward_cpu", 
		&intensity_transform_batch_forward_cpu, 
		"batch forward for intensity transform with cpu support", 
		py::arg("output"), 
		py::arg("image"), 
		py::arg("curve"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight")
	);
	m.def(
		"batch_backward_cpu", 
		&intensity_transform_batch_backward_cpu, 
		"batch backward for intensity transform with cpu support", 
		py::arg("curve_grad"), 
		py::arg("grad_from_output"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight")
	);
#ifdef WITH_CUDA
	m.def(
		"forward_cuda", 
		&intensity_transform_forward_cuda, 
		"forward for intensity transform with GPU support", 
		py::arg("output"), 
		py::arg("image"), 
		py::arg("curve"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight"),  
		py::arg("length"),
		py::arg("block_size")=128
	);
	m.def(
		"backward_cuda", 
		&intensity_transform_backward_cuda, 
		"backward for intensity transform with GPU support", 
		py::arg("curve_grad"), 
		py::arg("grad_from_output"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight"),  
		py::arg("length"),
		py::arg("block_size")=128
	);
	m.def(
		"batch_forward_cuda", 
		&intensity_transform_batch_forward_cuda, 
		"batch forward for intensity transform with GPU support", 
		py::arg("output"), 
		py::arg("image"), 
		py::arg("curve"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight")
	);
	m.def(
		"batch_backward_cuda", 
		&intensity_transform_batch_backward_cuda, 
		"batch backward for intensity transform with GPU support", 
		py::arg("curve_grad"), 
		py::arg("grad_from_output"), 
		py::arg("down_index"), 
		py::arg("up_index"),  
		py::arg("down_weight"),  
		py::arg("up_weight")
	);
#endif
}