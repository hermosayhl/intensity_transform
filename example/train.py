# Python
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')
# 3rd party
import cv2
import numpy
import dill as pickle
# torch
import torch
torch.set_num_threads(2)
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# self
import utils
import evaluate
import pipeline
import architecture


# 设置随机种子
utils.set_seed(212)
# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_tensor_type(torch.FloatTensor)


# ------------------------------- 定义超参等 --------------------------------------

# 参数
opt = lambda: None
# 训练参数
opt.use_cuda = True
opt.optimizer = torch.optim.Adam
opt.lr = 1e-3
opt.total_epochs = 400
opt.train_batch_size = 16
opt.valid_batch_size = 1
opt.test_batch_size = 1
# 实验参数
opt.save = True
opt.valid_interval = 1
opt.test_ratio = 0.1
opt.exp_name = "demo_batch"
opt.small_size = (224, 224)
opt.checkpoints_dir = os.path.join("./checkpoints/", opt.exp_name)
opt.dataset_dir = "D:/data/datasets/MIT-Adobe_FiveK/png"
# 可视化参数
opt.visualize_size = 1
opt.visualize_batch = 200
opt.visualize_dir = os.path.join(opt.checkpoints_dir, 'train_phase') 
# 创建一些文件夹
for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
os.makedirs(opt.visualize_dir, exist_ok=True)



# ------------------------------- 定义数据读取 --------------------------------------
train_images_list, valid_images_list, test_images_list = pipeline.get_images(opt)
print('\ntrain  :  {}\nvalid  :  {}\ntest  :  {}'.format(len(train_images_list), len(valid_images_list), len(test_images_list)))
print(train_images_list[:3])
print(valid_images_list[:3])
# train
train_dataset = pipeline.FiveKPairedDataset(train_images_list, train=True, small_size=opt.small_size)
train_loader = DataLoader(
	train_dataset, 
	batch_size=opt.train_batch_size, 
	shuffle=True,
	pin_memory=True,
	worker_init_fn=utils.worker_init_fn)
# valid
valid_dataset = pipeline.FiveKPairedDataset(valid_images_list, train=False)
valid_loader = DataLoader(
	valid_dataset,
	batch_size=opt.valid_batch_size,
	shuffle=False,
	pin_memory=True)



# ------------------------------- 定义网络结构 --------------------------------------
network = architecture.IntensityTransformNet(small_size=opt.small_size)
if(opt.use_cuda):
	network = network.cuda()


# ------------------------------- 定义优化器和损失函数等 --------------------------------------

# 损失函数
train_evaluator = evaluate.ImageEnhanceEvaluator(psnr_only=True)

# 优化器
optimizer = opt.optimizer(filter(lambda p: p.requires_grad, network.parameters()), lr=opt.lr, weight_decay=1e-5) 

# 学习率调整策略
def lr_decay(epoch):
	# 4e-3 * 0.00003 = 1e-4
	if(epoch > 30): return 0.025
	# 4e-3 * 0.05 = 5e-4
	elif(epoch > 12): return 0.125
	# 4e-3 * 0.25 = 1e-3
	elif(epoch >= 3): return 0.25
	# 4e-3 * 0.5 = 2e-3
	elif(epoch >= 2): return 0.5
	# 4e-3
	else: return 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay)


# 保存本次的训练设置
with open(os.path.join(opt.checkpoints_dir, "options.pkl"), 'wb') as file:
	pickle.dump({
		"opt": opt, 
		"train_images_list": train_images_list, 
		"valid_images_list": valid_images_list, 
		"test_images_list" : test_images_list, 
		"train_evaluator"  : train_evaluator, 
		"optimizer"        : optimizer, 
		"scheduler"        : scheduler}, 
	file)


# 损失函数
criterion = torch.nn.MSELoss()


# ------------------------------- 开始训练 --------------------------------------
for ep in range(1, opt.total_epochs + 1):
	print()
	# 计时验证的时间
	with utils.Timer() as time_scope:
		network.train()
		train_evaluator.clear()
		# 迭代 batch
		for train_batch, (low_quality, high_quality, image_name) in enumerate(train_loader, 1):
			# 清空梯度
			optimizer.zero_grad()
			# 数据送到 GPU
			if(opt.use_cuda):
				low_quality, high_quality = low_quality.cuda(non_blocking=True), high_quality.cuda(non_blocking=True)
			# 经过网络
			enhanced = network(low_quality).clamp_(0, 1)
			# 评估损失
			loss_value = criterion(enhanced, high_quality)
			# 损失回传
			loss_value.backward()
			# w -= lr * gradient
			optimizer.step()

			train_evaluator.update(enhanced, high_quality)
			# 输出信息
			output_infos = '\rTrain===> [epoch {}/{}] [batch {}/{}] [loss {:.3f}] [mse {:.4f}] [psnr {:.3f}] [lr {:.5f}]'.format(
				ep, opt.total_epochs, train_batch, len(train_loader), *train_evaluator.get(), optimizer.state_dict()['param_groups'][0]['lr'])
			sys.stdout.write(output_infos)
			# 可视化一些图像
			if(train_batch % opt.visualize_batch == 0 and opt.train_batch_size % opt.visualize_size == 0):
				utils.visualize_a_batch(enhanced, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_enhanced.png".format(ep, train_batch)))
				utils.visualize_a_batch(low_quality, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_low_quality.png".format(ep, train_batch)))
		# 更新学习率
		# scheduler.step()
	# --------------------------- validation ------------------------
	# 验证
	if(ep % opt.valid_interval == 0):
		with utils.Timer() as time_scope:
			network.eval()
			valid_evaluator = evaluate.ImageEnhanceEvaluator(psnr_only=True)
			with torch.no_grad():
				for valid_batch, (low_quality, high_quality, image_name) in enumerate(valid_loader, 1):
					# 数据送到 GPU
					if(opt.use_cuda):
						low_quality, high_quality = low_quality.cuda(non_blocking=True), high_quality.cuda(non_blocking=True)
					# 经过网络
					enhanced = network(low_quality)[0]
					enhanced = torch.clamp(enhanced, 0, 1)
					# 评估损失
					valid_evaluator.update(enhanced, high_quality)
					# 输出信息
					output_infos = '\rvalid===> [epoch {}/{}] [batch {}/{}] [loss {:.3f}] [mse {:.3f}] [psnr {:.3f}]'.format(
						ep, opt.total_epochs, valid_batch, len(valid_loader), *valid_evaluator.get())
					sys.stdout.write(output_infos)
				# 保存网络
				save_path = os.path.join(opt.checkpoints_dir, 
					'epoch_{}_train_{:.3f}_valid_{:.3f}.pth'.format(ep, train_evaluator.get()[2], valid_evaluator.get()[2]))
				print(' ---- saved to {}'.format(save_path), end="\t")
				torch.save(network.state_dict(), save_path)



				



