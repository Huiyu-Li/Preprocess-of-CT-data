import os
import time
import shutil
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
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
					  idx,origin,direction,xyz_thickness,savedct_path,savedseg_path):
	num_z = (ct_array.shape[0]-blockz)//stridez + 1#math.floor()
	num_x = (ct_array.shape[1]-blockx)//stridex + 1
	num_y = (ct_array.shape[2]-blocky)//stridey + 1
	for z in range(num_z):
		for x in range(num_x):
			for y in range(num_y):
				seg_block = seg_array[z*stridez:z*stridez+blockz,x*stridex:x*stridex+blockx,y*stridey:y*stridey+blocky]
				if seg_block.any():
					ct_block = ct_array[z * stridez:z * stridez + blockz, x * stridex:x * stridex + blockx,
							   y * stridey:y * stridey + blocky]
					saved_ctname = os.path.join(savedct_path, 'volume-' + str(idx) + '.nii')
					saved_segname = os.path.join(savedseg_path, 'segmentation-' + str(idx) + '.nii')
					saved_preprocessed(ct_block,origin,direction,xyz_thickness,saved_ctname)
					saved_preprocessed(seg_block,origin,direction,xyz_thickness,saved_segname)
					idx = idx + 1
	return idx

def saved_preprocessed(savedImg,origin,direction,xyz_thickness,saved_name):
	newImg = sitk.GetImageFromArray(savedImg)
	newImg.SetOrigin(origin)
	newImg.SetDirection(direction)
	newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
	sitk.WriteImage(newImg, saved_name)

def preprocess():
	start_time = time.time()
	##########hyperparameters1##########
	file_path = '/media/data/LITS/Training_dataset/'
	num_file = len(os.listdir(file_path))
	savedct_path = '/media/data/LITS/Preprocessed16_48/ct/'
	savedseg_path = '/media/data/LITS/Preprocessed16_48/seg/'
	expand_slice = 10
	blockz = 16;blockx = 48;blocky = 48
	stridez = blockz;stridex = blockx; stridey = blocky
	# stridez = blockz//3;stridex = blockx//3;stridey = blocky//3
	xyz_thickness = [1.0, 1.0, 1.0]
	saved_idx = 0
	##########end hyperparameters1##########
	# Clear saved dir
	if os.path.exists(savedct_path) is True:
		shutil.rmtree(savedct_path)
	os.mkdir(savedct_path)
	if os.path.exists(savedseg_path) is True:
		shutil.rmtree(savedseg_path)
	os.mkdir(savedseg_path)

	for i in range(1):#num_file
		ct = sitk.ReadImage(os.path.join(file_path,'volume-'+str(i)+'.nii'), sitk.sitkFloat32)# sitk.sitkInt16 Read one image using SimpleITK
		origin = ct.GetOrigin()
		direction = ct.GetDirection()
		ct_array = sitk.GetArrayFromImage(ct)
		seg = sitk.ReadImage(os.path.join(file_path,'segmentation-'+str(i)+'.nii'), sitk.sitkFloat32)
		seg_array = sitk.GetArrayFromImage(seg)
		print('-------','volume-'+str(i)+'.nii','-------')
		print('original space', np.array(ct.GetSpacing()))
		print('original shape',ct_array.shape)

		# step1: spacing interpolation
		# order=0:nearest interpolation;order=1:bilinear interpolation;order=3:cubic interpolation
		ct_array = ndimage.zoom(ct_array, (
		ct.GetSpacing()[-1] / xyz_thickness[-1], ct.GetSpacing()[0] / xyz_thickness[0],
		ct.GetSpacing()[1] / xyz_thickness[1]), order=3)
		# 对金标准插值不应该使用高级插值方式，这样会破坏边界部分,检查数据输出很重要！！！
		seg_array = ndimage.zoom(seg_array, (
		ct.GetSpacing()[-1] / xyz_thickness[-1], ct.GetSpacing()[0] / xyz_thickness[0],
		ct.GetSpacing()[1] / xyz_thickness[1]), order=0)
		print('new space', xyz_thickness)
		print('zoomed shape:', ct_array.shape, ',', seg_array.shape)

		# step2:window transform
		ct_array = window_transform(ct_array, windowWidth=200, windowCenter=40, normal=True)
		print('window transform:',ct_array.min(),ct_array.max())

		# step3:get mask effective range(startpostion:endpostion)
		z = np.any(seg_array, axis=(1, 2))  # seg_array.shape(125, 256, 256)
		start_slice, end_slice = np.where(z)[0][[0, -1]]
		if start_slice - expand_slice < 0:
			start_slice = 0
		else:
			start_slice -= expand_slice
		if end_slice + expand_slice >= seg_array.shape[0]:
			end_slice = seg_array.shape[0] - 1
		else:
			end_slice += expand_slice
		ct_array = ct_array[start_slice:end_slice + 1, :, :]
		seg_array = seg_array[start_slice:end_slice + 1, :, :]
		print('effective shape:', ct_array.shape,',',seg_array.shape)

		# step4:generate subimage
		# step5 save the preprocessed data
		saved_idx = generate_subimage(ct_array, seg_array, stridez, stridex, stridey, blockz, blockx, blocky,
						  saved_idx, origin, direction,xyz_thickness,savedct_path,savedseg_path)

		print('Time {:.3f} min'.format((time.time() - start_time) / 60))
		print(saved_idx)


def check_empty(savedseg_path):
	seg_lists = os.listdir(savedseg_path)
	num_file = len(seg_lists)
	for i in range(num_file):
		seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(savedseg_path, seg_lists[i])))
		if not seg_array.any():
			print(seg_lists[i])

if __name__ == '__main__':
	start_time = time.time()
	preprocess()
	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
