import sys
import os
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def Decide_preprocess(blockzxy,config):
	# Decide Preprocess of different stride, window wide-center
	# S2:9; S3:5; S4:3; S5:2 for suitble upper 100 and time saving
	# Decide stride 2 when wc = [],[150,75],[100,70]
	config['num_file'] = 9
	config['stridezxy'] = [blockzxy[0] // 2, blockzxy[1] // 2, blockzxy[2] // 2]
	config['window_wc'] = []
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_Wlivertumor/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_Wlivertumor/seg"
	preprocess(blockzxy, config)
	config['window_wc'] = [150, 75]
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W15075/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W15075/seg"
	preprocess(blockzxy, config)
	config['window_wc'] = [100, 70]
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W10070/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S2_W10070/seg"
	preprocess(blockzxy, config)

	# Decide stride 4 when wc = [],[150,75],[100,70]
	config['num_file'] = 3
	config['stridezxy'] = [blockzxy[0] // 4, blockzxy[1] // 4, blockzxy[2] // 4]
	config['window_wc'] = []
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_Wlivertumor/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_Wlivertumor/seg"
	preprocess(blockzxy, config)
	config['window_wc'] = [150,75]
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W15075/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W15075/seg"
	preprocess(blockzxy, config)
	config['window_wc'] = [100,70]
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W10070/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S4_W10070/seg"
	preprocess(blockzxy, config)

	# Decide stride 5 when wc = [],[150,75],[100,70]
	config['num_file'] = 2
	config['stridezxy'] = [blockzxy[0] // 5, blockzxy[1] // 5, blockzxy[2] // 5]
	config['window_wc'] = []
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_Wlivertumor/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_Wlivertumor/seg"
	preprocess(blockzxy, config)
	config['window_wc'] = [150, 75]
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W15075/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W15075/seg"
	preprocess(blockzxy, config)
	config['window_wc'] = [100, 70]
	config['savedct_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W10070/ct"
	config['savedseg_path'] = "/data/lihuiyu/LiTS/Decide_Preprocessing/S5_W10070/seg"
	preprocess(blockzxy, config)

	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))

def histgram_plot_save(train_path, check_path):
	# polt gray histgram
	num = 131
	bins = 100
	pbar = tqdm(total=num)  # Initialise
	for i in range(num):
		ct_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(train_path, 'volume-' + str(i) + '.nii')))
		seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(train_path, 'segmentation-' + str(i) + '.nii')))

		seg_bg = seg_array == 0
		seg_liver = seg_array >= 1
		seg_tumor = seg_array == 2

		ct_bg = ct_array * seg_bg
		ct_liver = ct_array * seg_liver
		ct_tumor = ct_array * seg_tumor

		bg_min = ct_bg.min()
		bg_max = ct_bg.max()
		liver_min = ct_liver.min()
		liver_max = ct_liver.max()
		tumor_min = ct_tumor.min()
		tumor_max = ct_tumor.max()

		ct_bg = np.float32(ct_bg)
		ct_liver = np.float32(ct_liver)
		ct_tumor = np.float32(ct_tumor)
		hist_bg = cv2.calcHist([ct_bg.flatten()], [0], None, [bins], [int(bg_min), int(bg_max)])  # shape(100, 1)
		hist_liver = cv2.calcHist([ct_liver.flatten()], [0], None, [bins],
								  [int(liver_min), int(liver_max)])  # shape(100, 1)
		hist_tumor = cv2.calcHist([ct_tumor.flatten()], [0], None, [bins],
								  [int(tumor_min), int(tumor_max)])  # shape(100, 1)
		plt.figure()
		plt.plot(hist_bg, 'k')
		plt.plot(hist_liver, 'r')
		plt.plot(hist_tumor, 'g')
		plt.legend(('bg', 'liver', 'tumor'), loc='upper right')
		plt.title('Tissue Intensity Distribution' + 'volume-' + str(i))
		saved_name = os.path.join(check_path, 'volume-' + str(i) + '.png')
		plt.savefig(saved_name)
		pbar.update(1)
	pbar.close()

def get_GrayScaleRange(train_ct_path):
    # get the gray scale range of train-liver, train_tumor
    num = 131
    liver_lower = 0;liver_upper = 0
    tumor_lower = 0;tumor_upper = 0
    for i in range(num):
        ct_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(train_ct_path,'volume-'+str(i)+'.nii')))
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(train_ct_path,'segmentation-'+str(i)+'.nii')))
        seg_bg = seg_array==0
        seg_liver = seg_array>=1
        seg_tumor = seg_array==2

        ct_bg = ct_array*seg_bg
        ct_liver = ct_array*seg_liver
        ct_tumor = ct_array*seg_tumor

        liver_min = ct_liver.min()
        liver_max = ct_liver.max()
        tumor_min = ct_tumor.min()
        tumor_max = ct_tumor.max()
        print(i,'all:(',ct_array.min(),',',ct_array.max(),')',
              'bg:(',ct_bg.min(),',',ct_bg.max(),')',
              'liver:(',liver_min,',',liver_max,')',
              'tumor:(',tumor_min,',',tumor_max,')',)
        liver_lower+=liver_min
        liver_upper+=liver_max
        tumor_lower += tumor_min
        tumor_upper += tumor_max
    print('liver:(',liver_lower/num,',',liver_upper/num,')')
    print('tumor:(', tumor_lower / num, ',', tumor_upper / num, ')')


