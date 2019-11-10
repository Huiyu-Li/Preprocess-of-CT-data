import os
import time
import shutil
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from utils import *
import warnings
warnings.simplefilter("ignore")

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
	"""
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
	minWindow = float(windowCenter) - 0.5*float(windowWidth)
	newimg = (ct_array - minWindow) / float(windowWidth)
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	if not normal:
		newimg = (newimg * 255).astype('uint8')
	return newimg

def generate_subimage(ct_array,seg_array,stridez, stridex, stridey, blockz, blockx, blocky,
					  saved_idx,origin,direction,xyz_thickness,savedct_path,savedseg_path,saved_prefix):
	num_z = (ct_array.shape[0]-blockz)//stridez + 1#math.floor()
	num_x = (ct_array.shape[1]-blockx)//stridex + 1
	num_y = (ct_array.shape[2]-blocky)//stridey + 1
	idx = 0
	savedct_path = os.path.join(savedct_path,saved_prefix)
	savedseg_path = os.path.join(savedseg_path,saved_prefix.replace('volume','segmentation'))
	if os.path.exists(savedct_path)|os.path.exists(savedseg_path):
		shutil.rmtree(savedct_path)
		shutil.rmtree(savedseg_path)
	os.mkdir(savedct_path)
	os.mkdir(savedseg_path)
	for z in range(num_z):
		for x in range(num_x):
			for y in range(num_y):
				seg_block = seg_array[z*stridez:z*stridez+blockz,x*stridex:x*stridex+blockx,y*stridey:y*stridey+blocky]
				if seg_block.any():
					ct_block = ct_array[z * stridez:z * stridez + blockz, x * stridex:x * stridex + blockx,
							   y * stridey:y * stridey + blocky]
					saved_ctname = os.path.join(savedct_path,str(idx) +'.nii')
					saved_segname = os.path.join(savedseg_path,str(idx)+'.nii')
					saved_preprocessed(ct_block,origin,direction,xyz_thickness,saved_ctname)
					saved_preprocessed(seg_block,origin,direction,xyz_thickness,saved_segname)
					idx = idx + 1
	return saved_idx + idx

def saved_preprocessed(savedImg,origin,direction,xyz_thickness,saved_name):
	newImg = sitk.GetImageFromArray(savedImg)
	newImg.SetOrigin(origin)
	newImg.SetDirection(direction)
	newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
	sitk.WriteImage(newImg, saved_name)

def preprocess(blockzxy,config):
	# Clear saved dir
	if os.path.exists(config['savedct_path']) is True:
		shutil.rmtree(config['savedct_path'])
	os.makedirs(config['savedct_path'])
	if os.path.exists(config['savedseg_path']) is True:
		shutil.rmtree(config['savedseg_path'])
	os.makedirs(config['savedseg_path'])

	for i in range(config['num_file']):#num_file
		saved_prefix = 'volume-'+str(i)
		ct = sitk.ReadImage(os.path.join(config['file_path'],'volume-'+str(i)+'.nii'), sitk.sitkFloat32)# sitk.sitkInt16 Read one image using SimpleITK
		origin = ct.GetOrigin()
		direction = ct.GetDirection()
		ct_array = sitk.GetArrayFromImage(ct)
		seg = sitk.ReadImage(os.path.join(config['file_path'],'segmentation-'+str(i)+'.nii'), sitk.sitkFloat32)
		seg_array = sitk.GetArrayFromImage(seg)
		print('-------','volume-'+str(i)+'.nii','-------')
		print('original space', np.array(ct.GetSpacing()))
		print('original shape',ct_array.shape)

		# step1: spacing interpolation
		# order=0:nearest interpolation;order=1:bilinear interpolation;order=3:cubic interpolation
		ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
										   ct.GetSpacing()[0] / config['xyz_thickness'][0],
										   ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=3)
		# 对金标准插值不应该使用高级插值方式，这样会破坏边界部分,检查数据输出很重要！！！
		# 使用order=1可确保zoomed seg unique = [0,1,2]
		seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
											 ct.GetSpacing()[0] / config['xyz_thickness'][0],
											 ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=0)
		print('new space', config['xyz_thickness'])
		print('zoomed shape:', ct_array.shape, ',', seg_array.shape)

		# step2:window transform
		print('zoomed seg unique:',np.unique(seg_array))
		seg_tumor = seg_array == 2
		ct_tumor = ct_array * seg_tumor
		tumor_min = ct_tumor.min()
		tumor_max = ct_tumor.max()
		tumor_wide = tumor_max - tumor_min
		if config['window_wc']:
			# by customosed uniform window wide and center
			ct_array = window_transform(ct_array, config['window_wc'][0], config['window_wc'][1], normal=True)
			print('window transform(', config['window_wc'][0], ',', config['window_wc'][1], '):', ct_array.min(), ct_array.max())
		elif tumor_wide == 0:# handle the case which does not contain tumor
			# by liver
			seg_liver = seg_array >= 1
			ct_liver = ct_array * seg_liver
			liver_min = ct_liver.min()
			liver_max = ct_liver.max()
			liver_wide = liver_max - liver_min
			liver_center = (liver_max + liver_min) / 2
			ct_array = window_transform(ct_array, liver_wide, liver_center, normal=True)
			print('window transform(', liver_wide, ',', liver_center, '):', ct_array.min(), ct_array.max())
		else:
			# by tumor
			tumor_center = (tumor_max + tumor_min) / 2
			ct_array = window_transform(ct_array, tumor_wide, tumor_center, normal=True)
			print('window transform(', tumor_wide, ',', tumor_center, '):', ct_array.min(), ct_array.max())

		# step3:get mask effective range(startpostion:endpostion)
		z = np.any(seg_array, axis=(1, 2))  # seg_array.shape(125, 256, 256)
		start_slice, end_slice = np.where(z)[0][[0, -1]]
		if start_slice - config['expand_slice'] < 0:
			start_slice = 0
		else:
			start_slice -= config['expand_slice']
		if end_slice + config['expand_slice'] >= seg_array.shape[0]:
			end_slice = seg_array.shape[0] - 1
		else:
			end_slice += config['expand_slice']
		ct_array = ct_array[start_slice:end_slice + 1, :, :]
		seg_array = seg_array[start_slice:end_slice + 1, :, :]
		print('effective shape:', ct_array.shape,',',seg_array.shape)

		if ct_array.shape[0] < blockzxy[0]:
			print('generate no subimage !')
		else:
			# step4:generate subimage
			# step5 save the preprocessed data
			config['saved_idx'] = generate_subimage(ct_array, seg_array,
						config['stridezxy'][0], config['stridezxy'][1], config['stridezxy'][2],
						blockzxy[0], blockzxy[1], blockzxy[2],
						config['saved_idx'], origin, direction,config['xyz_thickness'],
						config['savedct_path'],config['savedseg_path'],saved_prefix)
		print(config['saved_idx'])

if __name__ == '__main__':
	start_time = time.time()
	logfile = './printLog'
	if os.path.isfile(logfile):
		os.remove(logfile)
	sys.stdout = Logger(logfile)#see utils.py
	##########hyperparameters##########
	blockzxy = [64, 256, 256]
	config = {
		'file_path' : "/data/lihuiyu/LiTS/Training_dataset",
		'savedct_path' : "/data/lihuiyu/LiTS/Preprocessed_S5_W20040/ct",
		'savedseg_path' : "/data/lihuiyu/LiTS/Preprocessed_S5_W20040/seg",
		'num_file' : 131,
		'window_wc':[200,40],#[]means by automatic liver and tumor center
		'stridezxy' : [blockzxy[0] // 5, blockzxy[1] // 5, blockzxy[2] // 5],
		'expand_slice' : 10,
		'xyz_thickness' : [1.0, 1.0, 1.0],
		'saved_idx' : 0
	}
	##########end hyperparameters#######
	# Normal preprocess
	print(config['savedct_path'].split('/')[-2])
	print(config['window_wc'])
	print(config['stridezxy'])
	preprocess(blockzxy,config)

	# Decide preprocess of different stride and window
	# Decide_preprocess(blockzxy,config)

	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
