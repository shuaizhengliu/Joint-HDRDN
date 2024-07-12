import os
import glob
import numpy as np
from ..utils import pack_raw_numpy
from scipy.io import savemat

cur_test_folder = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_dynamic_data/Integral_test' # modify your path
patch_size = 256
save_gtfolder = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/GT_crop_mat' # modify your path
test_list = os.listdir(cur_test_folder)
for scene in test_list:
    scene_name = os.path.splitext(scene)[0]
    print('scene_name', scene_name)
    scene_file = os.path.join(cur_test_folder, scene)
    scene_gt = np.load(scene_file)['hdr']
    c, h, w = scene_gt.shape
    n_h = h //patch_size
    n_w = w //patch_size
    tmp_h = n_h * patch_size
    tmp_w = n_w * patch_size
    scene_gt = scene_gt[:, :tmp_h, :tmp_w]
    result_raw = pack_raw_numpy(scene_gt, 'BGGR')
    savename = scene_name+'.mat'
    savemat(os.path.join(save_gtfolder, savename), {'rawhdr': result_raw})



