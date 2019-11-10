import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
import csv
import numpy as np
import time
import shutil
import sys
import os
import math
from visdom import Visdom
# my packages
from utils import *
import MyDataloader
# from TumorNetWithoutSource import *
from TumorNetAvgpoolRelu import *

#################initialization network##############
def weights_init(model):
	if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
		nn.init.kaiming_uniform_(model.weight.data, 0.25)
		nn.init.constant_(model.bias.data, 0)
	# elif isinstance(model, nn.InstanceNorm3d):
	# 	nn.init.constant_(model.weight.data,1.0)
	# 	nn.init.constant_(model.bias.data, 0)

def train_valid_seg(episode,config):
	# refresh save dir
	exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
	ckpt_dir = os.path.join(config['ckpt_dir'] + exp_id)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	logfile = os.path.join(ckpt_dir, 'log')
	sys.stdout = Logger(logfile)#see utils.py

	###############GPU,Net,optimizer,scheduler###############
	torch.manual_seed(0)
	if torch.cuda.is_available():
		net = TumorNet().cuda()#need to do this before constructing optimizer
		loss = DiceLoss().cuda()
	else:
		net = TumorNet()
		loss = DiceLoss()
	cudnn.benchmark = True  # True
	# net = DataParallel(net).cuda()
	# optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9,weight_decay=weight_decay)#SGD+Momentum
	# optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'],(0.9, 0.999),eps=1e-08,weight_decay=2e-4)
	optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#decay the learning rate after 100 epoches
	###############resume or initialize prams###############
	if config['if_test'] or config['if_resume']:
		print('if_test:',config['if_test'],'if_resume:',config['if_resume'])
		checkpoint = torch.load(config['model_dir'])
		net.load_state_dict(checkpoint)
	else:
		print('weight initialization')
		net.apply(weights_init)

	#test
	if config['if_test']:
		print('###################test###################')
		test_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['test_csv'],channels=config['CDHW'][0],
								depth=config['CDHW'][1], height=config['CDHW'][2],width=config['CDHW'][3]),
								batch_size=config['batchSize_TVT'][2], shuffle=False, pin_memory=True)
		saved_dir = test(test_loader, net, config['saved_dir'])
		return saved_dir

	# val_set_loader
	val_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['valid_csv'],channels=config['CDHW'][0],
							depth=config['CDHW'][1], height=config['CDHW'][2],width=config['CDHW'][3]),
							batch_size=config['batchSize_TVT'][1], shuffle=False,pin_memory=True)
	#################train-eval (epoch)##############################
	max_validtumor= 0.6
	for epoch in range(config['max_epoches']):
		for epi in range(episode):
			# train_set_loader
			train_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['train_csv_list'][epi], channels=config['CDHW'][0],
									depth=config['CDHW'][1], height=config['CDHW'][2],width=config['CDHW'][3]),
									batch_size=config['batchSize_TVT'][0], shuffle=True, pin_memory=True)
			print('######train epoch-epi', str(epoch),'-',str(epi), 'lr=', str(optimizer.param_groups[0]['lr']),'######')
			train_loss, train_tumor, train_iter = train(train_loader, net, loss, optimizer)
			scheduler.step(epoch)
			train_avgloss = sum(train_loss) / train_iter
			train_avgtumor = sum(train_tumor) / train_iter
			print("[%d-%d/%d], train_loss:%.4f, train_tumor:%.4f, Time:%.3fmin" %
				  (epoch, epi, config['max_epoches']-1, train_avgloss, train_avgtumor, (time.time() - start_time) / 60))

			print('######valid epoch-epi', str(epoch),'-',str(epi),'######')
			valid_loss, valid_tumor, valid_iter = validate(val_loader, net, loss, epoch, config['saved_dir'])
			valid_avgloss = sum(valid_loss) / valid_iter
			valid_avgtumor = sum(valid_tumor) / valid_iter
			print("[%d-%d/%d], valid_loss:%.4f, valid_tumor:%.4f, Time:%.3fmin " %
				  (epoch, epi, config['max_epoches'], valid_avgloss, valid_avgtumor, (time.time() - start_time) / 60))
			# print:lr,epoch/total,loss123,accurate,time

			#if-save-model:
			if max_validtumor < abs(valid_avgtumor):
				max_validtumor = abs(valid_avgtumor)
				state = {
					'epoche':epoch,
					'arch':str(net),
					'state_dict':net.state_dict(),
					'optimizer':optimizer.state_dict()
					#other measures
				}
				torch.save(state,ckpt_dir+'/checkpoint.pth.tar')
				#save model
				model_filename = ckpt_dir+'/model_'+str(epoch)+'-'+str(epi)+'-'+str(max_validtumor)[2:6]+'.pth'
				torch.save(net.state_dict(),model_filename)
				print('Model saved in',model_filename)
			viz.line([train_avgloss], [epoch * episode + epi], win='train', opts=dict(title='train avgloss'),update='append')
			viz.line([valid_avgloss], [epoch * episode + epi], win='valid', opts=dict(title='valid avgloss'),update='append')
			viz.line([valid_avgtumor], [epoch * episode + epi], win='tumor', opts=dict(title='valid avgtumor'),update='append')

def train(data_loader, net, loss, optimizer):
	net.train()#swithch to train mode
	epoch_loss = []
	epoch_tumor = []
	total_iter = len(data_loader)
	for i, (data,target,origin,direction,space,prefix,subNo) in enumerate(data_loader):
		if torch.cuda.is_available():
			data = data.cuda()
			target = target.cuda()
		output = net(data)
		loss_output = loss(output, target)
		tumor_dice = Dice(output, target)
		optimizer.zero_grad()#set the grade to zero
		loss_output.backward()
		optimizer.step()

		epoch_loss.append(loss_output.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
		epoch_tumor.append(tumor_dice)
		print("[%d/%d], loss:%.4f, tumor_dice:%.4f" % (i, total_iter, loss_output.item(),tumor_dice))
	return epoch_loss,epoch_tumor,total_iter

def validate(data_loader, net, loss, epoch, saved_dir):
	net.eval()
	epoch_loss = []
	epoch_tumor = []
	total_iter = len(data_loader)
	with torch.no_grad():#no backward
		for i, (data,target,origin,direction,space,prefix,subNo) in enumerate(data_loader):
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			output = net(data)
			loss_output = loss(output,target)
			tumor_dice = Dice(output, target)

			epoch_loss.append(loss_output.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
			epoch_tumor.append(tumor_dice)
			print("[%d/%d], loss:%.4f, tumor_dice:%.4f" % (i, total_iter, loss_output.item(), tumor_dice))

			if saved_dir and epoch % 10 == 0:
				saved_dir_new = os.path.join(saved_dir,prefix[0])
				if not os.path.exists(saved_dir_new):
					os.mkdir(saved_dir_new)

				output_name = os.path.join(saved_dir_new, subNo[0] + '.nii')
				saved_preprocessed(output, origin, direction, space, output_name)
				print(output_name)

	return epoch_loss, epoch_tumor, total_iter

def test(data_loader, net, saved_dir):
	net.eval()
	total_iter = len(data_loader)
	pbar = tqdm(total=total_iter) # Initialise
	older_name = ''
	with torch.no_grad():  # no backward
		for i, (data, target, origin, direction, space, ct_name) in enumerate(data_loader):
			if torch.cuda.is_available():
				data = data.cuda()
			output1, output2 = net(data)
			#saved as nii
			outer_name = ct_name[0].split('-')[0] + '-' + ct_name[0].split('-')[1]
			inter_name = ct_name[0].split('-')[2]
			if older_name != outer_name:
				older_name = outer_name
				new_saved_dir = os.path.join(saved_dir, outer_name)
				os.makedirs(new_saved_dir)
			output1_name = os.path.join(new_saved_dir, inter_name + '.nii')
			saved_preprocessed_testing(output1, origin, direction, space, output1_name)
			pbar.update(1)
			pbar.set_description("%s" % outer_name+'-'+inter_name)
	pbar.close()
	return saved_dir

def Automatic_Decide(config_splitData,episode,config):
    # Automatic_Decide appropriate stride,window,
    # common hyperparameters
    config['max_epoches'] = 30
    # # S2_W20040
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W20040/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W20040/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S2_W20040'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S2_W20040/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S2_W20040/valid.csv'
    # config['ckpt_dir'] = './results_S2_W20040/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S2_Wlivertumor
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_Wlivertumor/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_Wlivertumor/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S2_Wlivertumor'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S2_Wlivertumor/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S2_Wlivertumor/valid.csv'
    config['ckpt_dir'] = './results_S2_Wlivertumor/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S2_W15075
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W15075/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W15075/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S2_W15075'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S2_W15075/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S2_W15075/valid.csv'
    config['ckpt_dir'] = './results_S2_W15075/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S2_W10070
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W10070/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W10070/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S2_W10070'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S2_W10070/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S2_W10070/valid.csv'
    config['ckpt_dir'] = './results_S2_W10070/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S3_W20040
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_W20040/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_W20040/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S3_W20040'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S3_W20040/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S3_W20040/valid.csv'
    # config['ckpt_dir'] = './results_S3_W20040/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # # S3_Wlivertumor
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_Wlivertumor/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_Wlivertumor/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S3_Wlivertumor'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S3_Wlivertumor/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S3_Wlivertumor/valid.csv'
    # config['ckpt_dir'] = './results_S3_Wlivertumor/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config
    #
    # # S3_W15075
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_W15075/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_W15075/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S3_W15075'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S3_W15075/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S3_W15075/valid.csv'
    # config['ckpt_dir'] = './results_S3_W15075/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config
    #
    # # S3_W10070
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_W10070/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S3_W10070/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S3_W10070'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S3_W10070/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S3_W10070/valid.csv'
    # config['ckpt_dir'] = './results_S3_W10070/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S4_W20040
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W20040/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W20040/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S4_W20040'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S4_W20040/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S4_W20040/valid.csv'
    # config['ckpt_dir'] = './results_S4_W20040/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S4_Wlivertumor
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_Wlivertumor/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_Wlivertumor/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S4_Wlivertumor'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S4_Wlivertumor/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S4_Wlivertumor/valid.csv'
    config['ckpt_dir'] = './results_S4_Wlivertumor/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S4_W15075
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W15075/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W15075/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S4_W15075'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S4_W15075/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S4_W15075/valid.csv'
    config['ckpt_dir'] = './results_S4_W15075/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S4_W10070
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W10070/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W10070/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S4_W10070'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S4_W10070/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S4_W10070/valid.csv'
    config['ckpt_dir'] = './results_S4_W10070/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # # S5_W20040
    # config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W20040/ct"
    # config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W20040/seg"
    # config_splitData['TVTcsv'] = './TVTcsv_S5_W20040'
    # print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    # split_data_quicklyDecide(config_splitData)
    # config['train_csv_list'] = ['./TVTcsv_S5_W20040/train' + str(i) + '.csv' for i in range(episode)]
    # config['valid_csv'] = './TVTcsv_S5_W20040/valid.csv'
    # config['ckpt_dir'] = './results_S5_W20040/'
    # config['saved_dir'] = ''
    # train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S5_Wlivertumor
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_Wlivertumor/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S54_Wlivertumor/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S5_Wlivertumor'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S5_Wlivertumor/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S5_Wlivertumor/valid.csv'
    config['ckpt_dir'] = './results_S5_Wlivertumor/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S5_W15075
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W15075/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W15075/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S5_W15075'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S5_W15075/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S5_W15075/valid.csv'
    config['ckpt_dir'] = './results_S5_W15075/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

    # S5_W10070
    config_splitData['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W10070/ct"
    config_splitData['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W10070/seg"
    config_splitData['TVTcsv'] = './TVTcsv_S5_W10070'
    print('######split_data:', config_splitData['savedct_path'].split('/')[-2])
    split_data_quicklyDecide(config_splitData)
    config['train_csv_list'] = ['./TVTcsv_S5_W10070/train' + str(i) + '.csv' for i in range(episode)]
    config['valid_csv'] = './TVTcsv_S5_W10070/valid.csv'
    config['ckpt_dir'] = './results_S5_W10070/'
    config['saved_dir'] = ''
    train_valid_seg(episode, config)  # when parameters is tediously, just use config

if __name__ == '__main__':
	# print(torch.__version__)#0.4.1
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
	start_time = time.time()

	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	viz = Visdom(env='Decide Server68-GPU1-DecideResidual')
	viz.line([0], [0], win='train')
	viz.line([0], [0], win='valid')
	viz.line([0], [0], win='tumor')
	##########hyperparameters##########
	config_splitData = {
		'savedct_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040/ct",
		'savedseg_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040/seg",
		'TVTcsv' : './TVTcsv_S3_W20040',
		'valid_csv' : './valid.csv',
		# test_csv = 'test.csv',
		'episode' : 1,
		'ratio' : 0.9
	}
	##########end hyperparameters##########
	##########hyperparameters##########
	episode = 1
	config = {
		'if_test' : False,
		'if_resume' : True,
		'max_epoches' : 100,
		'batchSize_TVT' : [1, 1, 1], #batchSize of train_valid_test
		'CDHW' : [1, 64, 256, 256],
		'learning_rate' : 0.00001,
		'weight_decay' : 1e-4,
		# 'model':'USNETres',
		'train_csv_list': ['./TVTcsv_S3_W20040/train' + str(i) + '.csv' for i in range(episode)],
		'valid_csv': './TVTcsv_S3_W20040/valid.csv',
		'test_csv': './test.csv',
		# below is the saved path
		'ckpt_dir': './results_S3_W20040/',
		'saved_dir': "",
		'model_dir': "./results/model_0-9-8222.pth"
	}
	##########hyperparameters##########
	# split_data(config_splitData)
	# train_valid_seg(episode, config)

	# check stride, window
	Automatic_Decide(config_splitData, episode, config)

	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))