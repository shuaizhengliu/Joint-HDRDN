import os
from glob import glob
import numpy as np
import argparse

#data_dir = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_dynamic_data/Integral_test'
#save_dir = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_dynamic_data/Integral_test_p256_noborder'

#data_dir = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_tranlate_static_data/Maxdisp20_randint/Test_data'
#save_dir = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_tranlate_static_data/Maxdisp20_randint/Test_data_p256_noborder'

#data_dir = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/hdr_without_GT_npz'
#save_dir = '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/hdr_without_GT_npz_p256_noborder'

data_dir = '' # The dir of original test data of full size 
save_dir = '' # The dir for saving croped test data of patch size 256. 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path_all_npz = glob(os.path.join(data_dir, '*.npz'), recursive=True)
path_all_npz = sorted(path_all_npz)

patch_size = 256
for ii in range(len(path_all_npz)):
    sub_patch = 0
    if (ii+1) % 10 == 0:
        print('    The {:d} original images'.format(ii+1))
    npzpath = path_all_npz[ii]
    print(npzpath)
    npzname = os.path.basename(npzpath)
    scene_name = os.path.splitext(npzname)[0]
    arrays = np.load(npzpath,allow_pickle=True)
    sht = arrays['sht']
    mid = arrays['mid']
    lng = arrays['lng']
    hdr = arrays['hdr']
    data_stack = np.concatenate([sht, mid, lng, hdr])
    print(data_stack.shape, data_stack.dtype)

    c, h, w = data_stack.shape

    n_h = h // patch_size
    n_w = w // patch_size
    tmp_h = n_h * patch_size
    tmp_w = n_w * patch_size
    tmp_data = np.ones((c, tmp_h, tmp_w), dtype=np.float32)
    tmp_data = data_stack[:,:tmp_h, :tmp_w]
    sub_patch = 0
    for x in range(n_w):
        for y in range(n_h):
            if (x+1) * patch_size <= tmp_w and (y+1) * patch_size <= tmp_h:
                temp_patch = tmp_data[:, y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                pch_savename = '%s_%04d.npz' % (scene_name, sub_patch)
                pch_savepath = os.path.join(save_dir, pch_savename)
                print('save gt path', pch_savepath)
                np.savez(pch_savepath, sht = temp_patch[:8], mid = temp_patch[8:16], lng = temp_patch[16:24], hdr = temp_patch[24:28])
                sub_patch += 1
    assert sub_patch == n_w * n_h

print('Finish!\n')

