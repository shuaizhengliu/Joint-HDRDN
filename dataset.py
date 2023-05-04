import numpy as np
#import cv2
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
import torch
import sys
import os.path as osp


class DatasetNpy_Mix(Dataset):
    # Each patch is saved in .npy file.
    # .npy  datatype: normalized, and unified bayer pattern of BGGR
    #   ---['sht']  [0:4] ldr [4:8] hdr
    #   ---['mid']  [0:4] ldr [4:8] hdr
    #   ---['lng']  [0:4] ldr [4:8] hdr
    #   ---['hdr']  [0:4] hdr

    def __init__(self, data_folder_list, patch_size, training=True):
        self.data_folder_list = data_folder_list
        #print(data_folder)
        img_list = []
        for data_folder in self.data_folder_list:
            img_list = img_list + sorted(glob.glob(os.path.join(data_folder, '*.npz')))
        self.img_list = img_list
        self.patch_size = patch_size
        self.training = training

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        npypath = self.img_list[index]
        save_name = os.path.basename(npypath)
        save_name = os.path.splitext(save_name)[0]
        imdata = np.load(npypath)
        sht = imdata['sht'] # shape [8,H,W]
        mid = imdata['mid']
        lng = imdata['lng']
        hdr = imdata['hdr'] # shape [4,H,W]
        
        if self.training:
            # concat for preprocessing
            imstack = np.concatenate([sht, mid, lng, hdr], axis=0)
            # 1. random crop
            imstack = self.random_crop(imstack)
        
            # split to return
            sht = self.to_tensor(imstack[0:8, :, :])
            mid = self.to_tensor(imstack[8:16, :, :])
            lng = self.to_tensor(imstack[16:24, :, :])
            hdr = self.to_tensor(imstack[24:, :, :])

            return {'sht': sht, 'mid': mid, 'lng': lng, 'hdr': hdr}
        
        else:
            sht = self.to_tensor(sht)
            mid = self.to_tensor(mid)
            lng = self.to_tensor(lng)
            hdr = self.to_tensor(hdr)

            return {'sht': sht, 'mid': mid, 'lng': lng, 'hdr': hdr, 'save_name':save_name}     

    def to_tensor(self, np_array):
        t = torch.from_numpy(np_array).float()
        return t

    def random_crop(self, np_array):
        c, h, w = np_array.shape
        assert(c==28)
        w_start = random.randint(0, w - self.patch_size)
        h_start = random.randint(0, h - self.patch_size)
        crop_array = np_array[:, h_start: h_start+self.patch_size, w_start: w_start+self.patch_size]
        return crop_array


if __name__ == '__main__':
    folderpath = ''
    dataset = DatasetNpy(data_folder = folderpath, patch_size = 128)
    train_loader = DataLoader(dataset = dataset, batch_size = 16, shuffle=True, num_workers=8)
    for (i, cur_data) in enumerate(train_loader):
        sht = cur_data['sht']
        mid = cur_data['mid']        
        lng = cur_data['lng']
        hdr = cur_data['hdr']
        print(sys.getsizeof(sht) , 'MB')
        print(sys.getsizeof(mid) / 1024 / 1024, 'MB')
        print(sys.getsizeof(lng) / 1024 / 1024, 'MB')
        
        print('{}:{}, {}, {}, {}'.format(i, sht.shape, mid.shape, lng.shape, hdr.shape))


        

