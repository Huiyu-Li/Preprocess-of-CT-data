# Preprocess.py

## hyperparameters in preprocess()

##########hyperparameters1##########
file_path = the dir of raw data
num_file = len(os.listdir(file_path))
savedct_path = the dir to saved the preprocessed ct image
savedseg_path = the dir to saved the preprocessed label
expand_slice = the expantasion slice of effective ct range
blockz ;blockx;blocky = the bloch size of zxy
stridez ;stridex;stridey = the stride of subimage generator
xyz_thickness = the new spacing
saved_idx = the index of saved image
##########end hyperparameters1##########

## Step in Preprocess

#### step1: spacing interpolation

#### step2:window transform

#### step3:get mask effective range

#### step4:generate subimage

#### step5 save the preprocessed data