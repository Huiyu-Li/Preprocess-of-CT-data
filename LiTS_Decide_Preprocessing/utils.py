import csv
import math
import numpy as np
import os
import sys
import shutil
# import matplotlib
# matplotlib.use('agg')

# split data
import re
def atoi(s):
	return int(s) if s.isdigit() else s

def natural_keys(text):
	return [atoi(c) for c in re.split('(\d+)', text)]

def get_totalList(file_path):
    # get the full list of sub files
    calculator = 0
    total_upper = 100  # to restrict the number of traing files
    full_list = []
    file_list = os.listdir(file_path)
    num = len(file_list)
    for i in range(num):
        prefix = 'volume-' + str(i)
        sub_list = os.listdir(os.path.join(file_path, prefix))
        sub_list.sort(key=natural_keys)
        sub_num = len(sub_list)
        calculator += len(sub_list)
        for j in range(sub_num):
            full_list.extend([os.path.join(file_path, prefix,sub_list[j])])

        if calculator >= total_upper:
            return full_list
    return full_list

def split_data_quicklyDecide(config):
	#Split data into epi-train-valid-test
	ct_lists = get_totalList(config['savedct_path'])#when quickly dicide stride and window
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for ct_name in train_lists:
				seg_name = ct_name.replace('volume', 'segmentation').replace('ct', 'seg')
				w.writerow((ct_name, seg_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for ct_name in valid_lists:
			seg_name = ct_name.replace('volume', 'segmentation').replace('ct', 'seg')
			w.writerow((ct_name, seg_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def split_data(config):
	#Split data into epi-train-valid-test
	ct_lists = os.listdir(config['savedct_path'])
	ct_lists.sort(key=natural_keys)
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for name in train_lists:
				# sub dir
				sub_list = os.listdir(os.path.join(config['savedct_path'], name))
				sub_list.sort(key=natural_keys)
				for sub in sub_list:
					ct_name = os.path.join(config['savedct_path'], name, sub)
					seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub)
					w.writerow((ct_name, seg_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for name in valid_lists:
			# sub dir
			sub_list = os.listdir(os.path.join(config['savedct_path'], name))
			sub_list.sort(key=natural_keys)
			for sub in sub_list:
				ct_name = os.path.join(config['savedct_path'], name, sub)
				seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub)
				w.writerow((ct_name, seg_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def getFreeId():
    import pynvml

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput == 'all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu ' + 'nx-x' + 'is being used') #//ValueError('gpu ' + g + 'is being used')
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message) #print to screen
        self.log.write(message) #print to logfile

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass