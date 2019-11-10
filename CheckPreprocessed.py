# saved the preporcessed results as png and check quickly them
import os
import shutil
import cv2
import time
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
from utils import *

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

# check the window_level
def check_window_level(ct_path,check_path):
    # check the result of different window wide and center
    if os.path.exists(check_path):
        shutil.rmtree(check_path)
    os.mkdir(check_path)

    for i in range(131):
        ct = sitk.ReadImage(os.path.join(ct_path, 'volume-' + str(i) + '.nii'))
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_path, 'segmentation-' + str(i) + '.nii')))
        slice = ct_array.shape[0] // 3 *2

        seg_bg = seg_array==0
        seg_liver = seg_array >= 1
        seg_tumor = seg_array == 2

        ct_bg = ct_array * seg_bg
        ct_liver = ct_array * seg_liver
        ct_tumor = ct_array * seg_tumor

        liver_min = ct_liver.min()
        liver_max = ct_liver.max()
        tumor_min = ct_tumor.min()
        tumor_max = ct_tumor.max()
        bg_min = ct_bg.min()
        bg_max = ct_bg.max()

        liver_wide = liver_max - liver_min
        liver_center = (liver_max + liver_min) / 2

        tumor_wide = tumor_max - tumor_min
        tumor_center = (tumor_max + tumor_min) / 2
        if tumor_wide == 0:
            # by liver
            liver_wl = window_transform(ct_array, liver_wide, liver_center, normal=True)
            plt.figure()
            plt.imshow(liver_wl[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path, 'volume-' + str(i) + '-' + str(liver_wide) + '_' + str(liver_center) + '.png')
            plt.savefig(saved_name)

            liver_200_40 = window_transform(ct_array, 200, 40, normal=True)
            plt.figure()
            plt.imshow(liver_200_40[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path,'volume-' + str(i) + '-200_40.png')
            plt.savefig(saved_name)

            liver_150_75 = window_transform(ct_array, 150, 75, normal=True)
            plt.figure()
            plt.imshow(liver_150_75[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path, 'volume-' + str(i) + '-150_75.png')
            plt.savefig(saved_name)

            liver_100_70 = window_transform(ct_array, 100, 70, normal=True)
            plt.figure()
            plt.imshow(liver_100_70[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path, 'volume-' + str(i) + '-100_70.png')
            plt.savefig(saved_name)

            print(str(i),'bg(',bg_min,',', bg_max,')liver(', liver_min, ',', liver_max, ')tumor(',tumor_min,',',tumor_max,')')
            print('liver_wc(', liver_wide, ',', liver_center, ')tumor_wc(',tumor_wide,',',tumor_center,')')
        else:
            # by tumor
            tumor_wl = window_transform(ct_array, tumor_wide, tumor_center, normal=True)
            plt.figure()
            plt.imshow(tumor_wl[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path, 'volume-' + str(i) + '-' + str(tumor_wide) + '_' + str(tumor_center) + '.png')
            plt.savefig(saved_name)

            liver_200_40 = window_transform(ct_array, 200, 40, normal=True)
            plt.figure()
            plt.imshow(liver_200_40[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path,'volume-' + str(i) + '-200_40.png')
            plt.savefig(saved_name)

            liver_150_75 = window_transform(ct_array, 150, 75, normal=True)
            plt.figure()
            plt.imshow(liver_150_75[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path, 'volume-' + str(i) + '-150_75.png')
            plt.savefig(saved_name)

            liver_100_70 = window_transform(ct_array, 100, 70, normal=True)
            plt.figure()
            plt.imshow(liver_100_70[slice, :, :], cmap='gray')
            saved_name = os.path.join(check_path, 'volume-' + str(i) + '-100_70.png')
            plt.savefig(saved_name)

            print(str(i), 'bg(', bg_min, ',', bg_max, ')liver(', liver_min, ',', liver_max, ')tumor(', tumor_min, ',',tumor_max,')')
            print('liver_wc(', liver_wide, ',', liver_center, ')tumor_wc(', tumor_wide, ',', tumor_center, ')')

# check the order of zoom_window
def window_level_case(ct_array,seg_array):
    seg_tumor = seg_array == 2
    ct_tumor = ct_array * seg_tumor
    tumor_min = ct_tumor.min()
    tumor_max = ct_tumor.max()
    tumor_wide = tumor_max - tumor_min
    if tumor_wide == 0:
        #by liver
        seg_liver = seg_array >= 1
        ct_liver = ct_array * seg_liver
        liver_min = ct_liver.min()
        liver_max = ct_liver.max()
        liver_wide = liver_max - liver_min
        liver_center = (liver_max + liver_min) / 2
        wl = window_transform(ct_array, liver_wide, liver_center, normal=True)
    else:
        #by tumor
        tumor_center = (tumor_max + tumor_min) / 2
        wl = window_transform(ct_array, tumor_wide, tumor_center, normal=True)
    return wl

def check_zoom_window(ct_path,check_path):
    #check the order of zoom and window transform
    if os.path.exists(check_path):
        shutil.rmtree(check_path)
    os.mkdir(check_path)
    xyz_thickness = [1.0, 1.0, 1.0]
    pbar = tqdm(total=131)  # Initialise
    for i in range(131):
        ct = sitk.ReadImage(os.path.join(ct_path, 'volume-' + str(i) + '.nii'))
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_path, 'segmentation-' + str(i) + '.nii')))
        slice = ct_array.shape[0] // 3 * 2

        # zoom before than window
        ct_array_zoom = ndimage.zoom(ct_array, (
            ct.GetSpacing()[-1] / xyz_thickness[-1], ct.GetSpacing()[0] / xyz_thickness[0],
            ct.GetSpacing()[1] / xyz_thickness[1]), order=3)
        seg_array_zoom = ndimage.zoom(seg_array, (
            ct.GetSpacing()[-1] / xyz_thickness[-1], ct.GetSpacing()[0] / xyz_thickness[0],
            ct.GetSpacing()[1] / xyz_thickness[1]), order=0)
        ct_array_zoom = window_level_case(ct_array_zoom,seg_array_zoom)
        plt.figure()
        plt.imshow(ct_array_zoom[slice, :, :], cmap='gray')
        saved_name = os.path.join(check_path,'volume-' + str(i) + 'zoom.png')
        plt.savefig(saved_name)

        # window before than zoom
        ct_array_window = window_level_case(ct_array, seg_array)
        ct_array_window = ndimage.zoom(ct_array_window, (
            ct.GetSpacing()[-1] / xyz_thickness[-1], ct.GetSpacing()[0] / xyz_thickness[0],
            ct.GetSpacing()[1] / xyz_thickness[1]), order=3)
        plt.figure()
        plt.imshow(ct_array_window[slice, :, :], cmap='gray')
        saved_name = os.path.join(check_path,'volume-' + str(i) + 'window.png')
        plt.savefig(saved_name)

        pbar.update(1)
    pbar.close()

# check generate_subimage
def generate_subimage(ct_array,seg_array,stridez, stridex, stridey, blockz, blockx, blocky,
					  saved_path,ct_name,seg_name):
    num_z = (ct_array.shape[0]-blockz)//stridez + 1#math.floor()
    num_x = (ct_array.shape[1]-blockx)//stridex + 1
    num_y = (ct_array.shape[2]-blocky)//stridey + 1
    print(num_y)
    plt.figure()
    for z in range(num_z):
        for x in range(num_x):
            for y in range(min(num_y,4)):
                seg_block = seg_array[z*stridez:z*stridez+blockz,x*stridex:x*stridex+blockx,y*stridey:y*stridey+blocky]
                if seg_block.any():
                    ct_block = ct_array[z * stridez:z * stridez + blockz, x * stridex:x * stridex + blockx,
                               y * stridey:y * stridey + blocky]
                    # the first 4 blocks
                    plt.subplot(1,4,y+1)
                    plt.axis('off')
                    plt.imshow(ct_block[30, :, :], cmap='gray')
            saved_name = os.path.join(saved_path, ct_name)
            plt.savefig(saved_name)
            return
def check_stride(ct_path,check_path):
    #check the appropriate stride of subimgae generator
    if os.path.exists(check_path):
        shutil.rmtree(check_path)
    os.mkdir(check_path)

    blockz = 64;blockx = 256;blocky = 256
    pbar = tqdm(total=131)  # Initialise
    for i in range(1):
        ct = sitk.ReadImage(os.path.join(ct_path, 'volume-' + str(i) + '.nii'))
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_path, 'segmentation-' + str(i) + '.nii')))

        stridezxy = [blockz // 2, blockx // 2, blocky // 2]
        ct_name = 'volume-' + str(i)+'s2.png'
        seg_name = 'segmentation-' + str(i)+'s2.png'
        generate_subimage(ct_array, seg_array, stridezxy[0], stridezxy[1], stridezxy[2],
                                      blockz, blockx, blocky,check_path, ct_name, seg_name)

        stridezxy = [blockz // 3, blockx // 3, blocky // 3]
        ct_name = 'volume-' + str(i) + 's3.png'
        seg_name = 'segmentation-' + str(i) + 's3.png'
        generate_subimage(ct_array, seg_array, stridezxy[0], stridezxy[1], stridezxy[2],
                                      blockz, blockx, blocky,check_path, ct_name, seg_name)

        stridezxy = [blockz // 4, blockx // 4, blocky // 4]
        ct_name = 'volume-' + str(i) + 's4.png'
        seg_name = 'segmentation-' + str(i) + 's4.png'
        generate_subimage(ct_array, seg_array, stridezxy[0], stridezxy[1], stridezxy[2],
                          blockz, blockx, blocky, check_path, ct_name, seg_name)

        stridezxy = [blockz // 5, blockx // 5, blocky // 5]
        ct_name = 'volume-' + str(i) + 's5.png'
        seg_name = 'segmentation-' + str(i) + 's5.png'
        generate_subimage(ct_array, seg_array, stridezxy[0], stridezxy[1], stridezxy[2],
                          blockz, blockx, blocky, check_path, ct_name, seg_name)
        pbar.update(1)
    pbar.close()

def check_NoPatches(ct_path):
    # print the case whose effective Z-shape smaller than blockz
    # so generate NoPatches
    # check the total pathches of different stride
    blockz = 64;blockx = 256;blocky = 256
    expand_slice = 10
    for i in range(131):
        prefix = 'volume-' + str(i)
        ct = sitk.ReadImage(os.path.join(ct_path, 'volume-' + str(i) + '.nii'))
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_path, 'segmentation-' + str(i) + '.nii')))

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

        stridezxy = [blockz // 2, blockx // 2, blocky // 2]
        temp = generate_subNumber(ct_array, seg_array, stridezxy[0], stridezxy[1], stridezxy[2], blockz, blockx, blocky,prefix)

# check zoomed and windowed tissue intensity
def check_preprocessed_histgram(ct_path, seg_path, check_path):
    # plot gray histgram
    if os.path.exists(check_path):
        shutil.rmtree(check_path)
    os.mkdir(check_path)
    num = 1 #131
    bins = 100
    pbar = tqdm(total=num)  # Initialise
    for i in range(num):
        sub_num = len(os.listdir(os.path.join(ct_path, 'volume-' + str(i))))
        for j in range(sub_num):
            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_path,'volume-'+str(i),str(j)+'.nii')))
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(seg_path,'segmentation-'+str(i),str(j)+'.nii')))

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
            print(bg_min,bg_max)
            print(liver_min, liver_max)
            print(tumor_min, tumor_max)

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
            saved_name = os.path.join(check_path, 'volume-'+str(i)+'-'+str(j) + '.png')
            plt.savefig(saved_name)
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    start_time = time.time()
    sys.stdout = Logger('./printLog_checkwl')  # see utils.py

    ct_path = ""
    saved_path = ""
    check_path = "/data/lihuiyu/LiTS/Decide_Preprocessing/histgram_check/"
    # check_window_level(saved_path,check_path)
    # check_zoom_window(ct_path, check_path)
    # check_stride(ct_path, check_path)
    # check_NoPatches(ct_path)

    preprocessed_ct_path = "/data/lihuiyu/LiTS/Preprocessed_S3_W20040/ct/"
    preprocessed_seg_path = "/data/lihuiyu/LiTS/Preprocessed_S3_W20040/seg/"
    check_preprocessed_histgram(preprocessed_ct_path, preprocessed_seg_path, check_path)

    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
