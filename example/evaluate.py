# Python
import math
# 3rd party
import cv2
import numpy
# tensorflow
import torch
from torch.autograd import Variable
import torch.nn.functional as F


###########################################################################
#                                Metrics
###########################################################################


class ImageEnhanceEvaluator():

	def __init__(self, psnr_only=True):
		self.psnr_only = psnr_only
		# mse 损失函数
		self.mse_loss_fn = torch.nn.MSELoss()
		# 统计一些值
		self.mean_psnr = 0
		self.mean_ssim = 0
		self.mean_loss = 0
		self.mean_mse_loss = 0
		# 统计第几次
		self.count = 0
		# 根据 mse_loss 计算 psnr
		self.compute_psnr = lambda mse: 10 * torch.log10(1. / mse).item() if(mse > 1e-5) else 50



	def update(self, label_image, pred_image):
		# 计数 + 1
		self.count += 1
		# mse loss
		mse_loss_value = self.mse_loss_fn(label_image, pred_image)
		self.mean_mse_loss += mse_loss_value.item()

		psnr_value = self.compute_psnr(mse_loss_value)
		self.mean_psnr += psnr_value
		# 计算损失
		total_loss_value = 1.0 * mse_loss_value
		
		self.mean_loss += total_loss_value.item()
		return total_loss_value


	def get(self):
		if(self.count == 0):
			return 0
		if(self.psnr_only):
			return self.mean_loss / self.count, self.mean_mse_loss * (255 ** 2) / self.count, self.mean_psnr / self.count

	def clear(self):
		self.count = 0
		self.mean_psnr = self.mean_ssim = self.mean_mse_loss = self.mean_loss = self.mean_tv_loss = self.mean_color_loss = 0

