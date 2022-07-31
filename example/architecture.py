# torch
import torch
import intensitytransform


class IntensityTransformFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, image, curve, mul_255=True):
		# 记录存在梯度回传的张量的形状, 方便 backward 时分配空间
		ctx.curve_size = curve.size()

		# 定义几个辅助张量, 内存交给 python 释放, 处理交给 C++、CUDA
		image_size  = image.size()
		down_index  = torch.zeros(image_size).int().to(image.device)
		up_index    = torch.zeros(image_size).int().to(image.device)
		down_weight = torch.zeros(image_size).float().to(image.device)
		up_weight   = torch.zeros(image_size).float().to(image.device)

		# 定义输出
		output      = torch.zeros(image_size).float().to(image.device)

		# 对输入做规范, 目前只支持 0-1 和 256 个节点的映射
		values      = torch.clip(image * 255, 0, 255) if(mul_255) else torch.clip(image, 0, 1)

		# C++ 处理
		intensitytransform.forward(
			output, 
			values,
			curve,
			down_index, 
			up_index, 
			down_weight, 
			up_weight, 
			image.numel(),
			128
		)
		if(mul_255):
			output.div_(255)

		# 保存反向梯度计算需要的张量
		ctx.save_for_backward(down_index, up_index, down_weight, up_weight)

		# 返回
		return output


	@staticmethod
	def backward(ctx, grad_from_output):
		# 定义返回的梯度(只有 curve 存在梯度)
		curve_grad = torch.zeros(ctx.curve_size).to(grad_from_output.device)

		# 拿出跟下一层返回的梯度相关的张量
		down_index, up_index, down_weight, up_weight = ctx.saved_tensors

		# C++ 计算梯度
		intensitytransform.backward(
			curve_grad, 
			grad_from_output, 
			down_index,
			up_index, 
			down_weight, 
			up_weight,
			grad_from_output.numel(),
			128
		)

		# 返回梯度(与 forward 输入一一对应)
		return (None, curve_grad, None)


	@staticmethod
	def symbolic(g, image, curve):
		return g.op("custom::intensitytransform", image, curve)




class BatchedIntensityTransformFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, image, curve, mul_255=True):
		# 记录存在梯度回传的张量的形状, 方便 backward 时分配空间
		ctx.curve_size = curve.size()

		# 定义几个辅助张量, 内存交给 python 释放, 处理交给 C++、CUDA
		image_size  = image.size()
		down_index  = torch.zeros(image_size).int().to(image.device)
		up_index    = torch.zeros(image_size).int().to(image.device)
		down_weight = torch.zeros(image_size).float().to(image.device)
		up_weight   = torch.zeros(image_size).float().to(image.device)

		# 定义输出
		output      = torch.zeros(image_size).float().to(image.device)

		# 对输入做规范, 目前只支持 0-1 和 256 个节点的映射
		values      = torch.clip(image * 255, 0, 255) if(mul_255) else torch.clip(image, 0, 1)

		# C++ 处理
		intensitytransform.batch_forward(
			output, 
			values,
			curve,
			down_index, 
			up_index, 
			down_weight, 
			up_weight, 
		)
		if(mul_255):
			output.div_(255)

		# 保存反向梯度计算需要的张量
		ctx.save_for_backward(down_index, up_index, down_weight, up_weight)

		# 返回
		return output


	@staticmethod
	def backward(ctx, grad_from_output):
		# 定义返回的梯度(只有 curve 存在梯度)
		curve_grad = torch.zeros(ctx.curve_size).to(grad_from_output.device)

		# 拿出跟下一层返回的梯度相关的张量
		down_index, up_index, down_weight, up_weight = ctx.saved_tensors

		# C++ 计算梯度
		intensitytransform.batch_backward(
			curve_grad, 
			grad_from_output, 
			down_index,
			up_index, 
			down_weight, 
			up_weight,
		)

		# 返回梯度(与 forward 输入一一对应)
		return (None, curve_grad, None)


	@staticmethod
	def symbolic(g, image, curve):
		return g.op("custom::intensitytransform", image, curve)





class IntensityTransformNet(torch.nn.Module):
	def __init__(self, mapping_nodes=256, channels=3, small_size=(224, 224)):
		super(IntensityTransformNet, self).__init__()

		self.conv_1 = torch.nn.Conv2d(3, 32, kernel_size=7, stride=3, padding=1, padding_mode="replicate")
		self.relu_1 = torch.nn.ReLU()
		self.conv_2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1, padding_mode="replicate")
		self.relu_2 = torch.nn.ReLU()
		self.pool   = torch.nn.MaxPool2d((9, 9))
		self.fc     = torch.nn.Linear(1024, mapping_nodes * channels)
		self.sz     = small_size


	def forward(self, x):
		batch_n = x.size(0)
		channel = x.size(1)

		# 1. 下采样成低分辨率
		low_res = torch.nn.functional.interpolate(x, size=self.sz, mode="bilinear", align_corners=False)

		# 2. 提取特征
		features = self.conv_1(low_res)
		features = self.relu_1(features)
		features = self.conv_2(features)
		features = self.relu_2(features)
		features = self.pool(features)

		# 3. 全连接层回归近似 N x 3 x 256 条曲线
		features = features.view(batch_n, -1)
		curves   = self.fc(features).view(batch_n, channel, -1).contiguous()

		# 训练时, 用固定小尺寸; 测试或者导出模型时直接算原始高分辨率
		target   = low_res if(self.training) else x

		# 4. 映射
		output   = BatchedIntensityTransformFunction.apply(target, curves)

		# output2  = torch.stack(
		# 	[
		# 		torch.stack([IntensityTransformFunction.apply(img, cur) for (img, cur) in zip(target[i], curves[i])])
		# 		for i in range(batch_n)
		# 	]
		# )
		# print(torch.allclose(output, output2))

		return output





if __name__ == '__main__':

	device = torch.device("cuda:0")
	
	network = IntensityTransformNet().to(device)
	network.eval()

	input_tensor = torch.randn(1, 3, 1080, 1920).to(device)

	output = network(input_tensor)
	print("output  :  ", output.shape)
