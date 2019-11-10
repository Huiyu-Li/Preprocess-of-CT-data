import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import SimpleITK as sitk

class LiTSDataloader(Dataset):
    def __init__(self,dir_csv,channels,depth, height,width):
        """
        :param csv_file:path to all the images
        """
        self.image_dirs = pd.read_csv(dir_csv,header=None).iloc[:, :].values#from DataFrame to array
        self.channels = channels
        self.image_depth = depth
        self.image_height = height
        self.image_width = width

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        # (NCDHW),C must be added and N must be removed since added byitself
        ct_array = np.empty((self.channels, self.image_depth, self.image_height,self.image_width)).astype(np.float32)
        seg_array = np.empty((self.channels, self.image_depth, self.image_height, self.image_width)).astype(np.float32)
        ct = sitk.ReadImage(self.image_dirs[item][0])
        ct_array[0, :, :, :] = sitk.GetArrayFromImage(ct)
        seg_array[0, :, :, :] = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[item][1]))
        origin =ct.GetOrigin()
        direction = ct.GetDirection()
        space = ct.GetSpacing()
        prefix = self.image_dirs[item][0].split('/')[-2]
        subNo = self.image_dirs[item][0].split('/')[-1][:-4]
        # sample = {'image':item_image,'label':item_label}#false code!
        return (ct_array,seg_array,origin,direction,space,prefix,subNo)