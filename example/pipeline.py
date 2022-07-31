import os
import cv2
import random
import numpy
import torch
import dill as pickle
from torch.utils.data import Dataset



def show(image, name='yhl'):
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def cv2_rotate(image, angle=15):
	height, width = image.shape[:2]    
	center = (width / 2, height / 2)   
	scale = 1                        
	M = cv2.getRotationMatrix2D(center, angle, scale)
	image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
	return image_rotation



def make_augment(low_quality, high_quality):
	# 以 0.9 的概率作数据增强
	if(random.random() > 1 - 0.9):
		# 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
		all_states = ['crop', 'flip', 'rotate']
		# 打乱增强的顺序
		random.shuffle(all_states)
		for cur_state in all_states:
			if(cur_state == 'flip'):
				# 0.5 概率水平翻转
				if(random.random() > 0.5):
					low_quality = cv2.flip(low_quality, 1)
					high_quality = cv2.flip(high_quality, 1)
					# print('水平翻转一次')
			elif(cur_state == 'crop'):
				# 0.5 概率做裁剪
				if(random.random() > 1 - 0.8):
					H, W, _ = low_quality.shape
					ratio = random.uniform(0.75, 0.95)
					_H = int(H * ratio)
					_W = int(W * ratio)
					pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
					low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					# print('裁剪一次')
			elif(cur_state == 'rotate'):
				# 0.2 概率旋转
				if(random.random() > 1 - 0.1):
					angle = random.randint(-15, 15)  
					low_quality = cv2_rotate(low_quality, angle)
					high_quality = cv2_rotate(high_quality, angle)
					# print('旋转一次')
	return low_quality, high_quality






class FiveKPairedDataset(Dataset):
	def __init__(self, images_list, train=False, small_size=(224, 224)):
		self.images_list = images_list
		self.is_train = train

		# .div(255)
		self.transform  = lambda x: torch.from_numpy(x).permute(2, 0, 1).type(torch.FloatTensor)
		# 16, 3, 256, 256 -> 16, 256, 256, 3
		self.restore    = lambda x: x.detach().cpu().mul(255).permute(0, 2, 3, 1).numpy().astype('uint8')

		self.small_size = small_size


	def __len__(self):
		return len(self.images_list)


	def __getitem__(self, idx):

		# 读取图像
		input_path, label_path = self.images_list[idx]
		low_quality = cv2.imread(input_path) * 1. / 255
		high_quality = cv2.imread(label_path) * 1. / 255

		# train
		if(self.is_train == True):
			# 数据增强
			low_quality, high_quality = make_augment(low_quality, high_quality)
			# 大小固定
			low_quality = cv2.resize(low_quality, self.small_size)
			high_quality = cv2.resize(high_quality, self.small_size)

		return self.transform(low_quality), self.transform(high_quality), os.path.split(input_path)[-1]





def get_images(opt):
	all_images_paths = [(os.path.join(opt.dataset_dir, 'input', it), os.path.join(opt.dataset_dir, 'expertC_gt', it)) for it in os.listdir(os.path.join(opt.dataset_dir, 'input'))]

	# 打乱数据集
	random.shuffle(all_images_paths)

	# 划分数据集
	total_size = len(all_images_paths)
	test_size  = int(total_size * opt.test_ratio)
	valid_size = test_size
	train_size = total_size - test_size - valid_size

	return all_images_paths[:train_size], \
		   all_images_paths[train_size: train_size + valid_size], \
		   all_images_paths[-test_size:]


