# Preprocess.py

## hyperparameters in preprocess()
All the hyperameters are in the config dict, which can provied convenience for you to change them
##########hyperparameters##########
blockzxy = the bloch size of zxy
config = {
    'file_path' : the dir of raw data,
    'savedct_path' : the dir to saved the preprocessed ct image,
    'savedseg_path' : the dir to saved the preprocessed label,
    'num_file' : number of files need to be preprocessed,
    'window_wc': the window wide and window center value of window transform. []means by automatic liver and tumor center
    'stridezxy' : the stride of subimage generator,
    'expand_slice' : the expantasion slice of effective ct range,
    'xyz_thickness' : the new spacing,
    'saved_idx' : calculate the total number of preproceed files
}
##########end hyperparameters#######

## Step in Preprocess

#### step1: spacing interpolation

#### step2:window transform

#### step3:get mask effective range

#### step4:generate subimage

#### step5 save the preprocessed data
> why we preprocess the ct images in this order?
we decide the order and appropriate value for each hyperparameter carefully by thorough experiemnts.
> we have found that:
##### Spacing interpolation:
should be the first step
Only order=1 can be used for seg interpolation, which can make sure shat unique(seg) = [0,1,2]
##### Window transform:
Should be checked carefully
So to find the appropriate window wide and center
Hanlde the exception that no tumor exists by using liver wide and center
Different window wide and center can be employed as data augmentation
##### Effective range:
When effenctive shapez < blockz, there is no generated subimage
##### Subimage generation:
Check the totoal number of patches and dicide which one to be used
Different stride may be employed as data augmentation

> If you want to find the appropriate experiment setup by yourself, you can run the code as necessary:

get_GrayScaleRange(train_ct_path) # get the gray scale range of train-liver, train_tumor, which can help you to inspect the intensity ralationshape between them.

CheckPreprocessed.py : Check the result of each step carefully(just plot, so easy but significant)

LiTS_Decide_Preprocessing： # Decide Preprocess of different stride, window wide-center