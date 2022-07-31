from setuptools import setup
from setuptools.command.build_ext import build_ext
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension



ext_name = "intensitytransform"

if(torch.cuda.is_available()):
	extension_type = CUDAExtension
	define_macros = [("WITH_CUDA", None)]
	source_files = [
		"cuda_extensions/intensity_transform.cu",
		"cuda_extensions/intensity_transform_batch_forward.cu",
		"cuda_extensions/intensity_transform_batch_backward.cu", 
		"cpp_extensions/intensity_transform.cpp",
	]
	print(f"Compiling with CUDA support")
else:
	extension_type = CppExtension
	define_macros = []
	source_files = [
		"cpp_extensions/intensity_transform.cpp"
	]


setup(
	name=ext_name, 
	version="0.1",
	author="liuchang-bupt",
	ext_modules=[
		extension_type(
			name=ext_name,
			sources=source_files,
			define_macros=define_macros
		)
	],
	cmdclass={
		"build_ext": BuildExtension
	}
)


# 1. Windows Visual Studio 2019 + CUDA10.1, 倘若发生错误 
# 	error: function "torch::OrderedDict<Key, Value>::Item::operator=(const torch::OrderedDict<std::string, at::Tensor>::Item &) [with Key=std::string, Value=at::Tensor]" (declared implicitly) cannot be referenced -- it is a deleted function
# 参考, 修改源码即可编译过去 https://github.com/pytorch/pytorch/pull/55275/files/bc4867bb5a074d810c6c5f233b47275322d875d0