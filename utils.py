import gc
import os
import sys

import numpy as np
from PIL import Image
#from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class train_data():
    def __init__(self, filepath='./data/image_clean_pat.npy'):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_clean_pat.npy'):
    return train_data(filepath=filepath)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data

def load_images_RGB(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('RGB')
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file).convert('RGB')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')

def save_images_RGB(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('RGB')
    im.save(filepath, 'png')


def save_images1(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    print(np.shape(ground_truth), np.shape(noisy_image), np.shape(clean_image))
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

'''
def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
'''

def raw_psnr(im1, im2, data_range):
    im1 = im1.astype(np.float)
    im2 = im2.astype(np.float)
    psnr = compare_psnr(im1[:, :, :], im2[:, :, :], data_range = data_range)
    return psnr

def batch_rawPSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = []
    for i in range(Img.shape[0]):
        psnr = compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        if np.isinf(psnr):
            continue
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)

def split_bayer(array):
    #for the array #block height width
    #attention! bayer split setting should be corresponding with that in test dataset
    bayer1 = array[:, ::2, ::2]
    bayer2 = array[:, ::2, 1::2]
    bayer3 = array[:, 1::2, ::2]
    bayer4 = array[:, 1::2, 1::2]
    return [bayer1, bayer2, bayer3, bayer4]


def mu_tonemap(x, mu=5000):
    return torch.log(1 + mu * x) / np.log(1 + mu)

def np_mu_tonemap(x, mu=5000):
    return (np.log(1 + mu * x)) / np.log(1 + mu)

def pack_raw(result_im, bayerpattern):
    # Input: result tensor of B x C x H x W, arraged as R-G-G-B
    # Output: the packeed raw tensor as indicated bayerpattern, shape is B x H x W
    bs,chan,h,w = result_im.shape 
    H, W = h*2, w*2
    img2 = torch.zeros((bs,H,W))
    if bayerpattern == 'BGGR':
        img2[:,0:H:2,0:W:2]=result_im[:,3,:,:]
        img2[:,0:H:2,1:W:2]=result_im[:,1,:,:]
        img2[:,1:H:2,0:W:2]=result_im[:,2,:,:]
        img2[:,1:H:2,1:W:2]=result_im[:,0,:,:]
    img2 = img2.squeeze(1).numpy()
    return img2

def pack_raw_numpy(result_im, bayerpattern):
    # Input: result numpy array of C x H x W, arraged as R-G-G-B
    # Output: the packeed raw array as indicated bayerpattern, shape is H x W
    chan,h,w = result_im.shape 
    H, W = h*2, w*2
    img2 = np.zeros((H,W))
    if bayerpattern == 'BGGR':
        img2[0:H:2,0:W:2]=result_im[3,:,:]
        img2[0:H:2,1:W:2]=result_im[1,:,:]
        img2[1:H:2,0:W:2]=result_im[2,:,:]
        img2[1:H:2,1:W:2]=result_im[0,:,:]
    return img2

def batch_rawSSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = []
    for i in range(Img.shape[0]):
        ssim = structural_similarity(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range, channel_axis=0)
        if np.isinf(ssim):
            continue
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM)


def test_single_img(model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=1, shuffle=False)
    # model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader)):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                 batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            output = model(batch_ldr0, batch_ldr1, batch_ldr2)
            img_dataset.update_result(torch.squeeze(output.detach().cpu()).numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    scene_name = img_dataset.scene_name
    return pred, label, scene_name


def rebuild(patch_npy_list, h, w, c):
    #given the h w of gt, assemble the patch result into an integral bayer data.
    #each npyfile's shape [c, hp, wp]
    #print(h,w,c)
    #print('length of npylist', len(patch_npy_list))
    patch_size = 256 # check the data setting in Make_data
    n_h = h //patch_size
    n_w = w //patch_size
    tmp_h = n_h * patch_size
    tmp_w = n_w * patch_size
    pred = np.empty((c, tmp_h, tmp_w), dtype=np.float32)
    for x in range(n_w):
        for y in range(n_h):
            selected_index = x*n_h+y
            #print(selected_index)
            patch_file = patch_npy_list[selected_index]
            cur_patch = np.load(patch_file)
            pred[:, y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = cur_patch
    return pred, tmp_h, tmp_w








    


'''
if __name__ == '__main__':
    noisyraw_path = '/home/lsz/Downloads/ValidationNoisyBlocksRaw(2).mat'
    gtraw_path = '/home/lsz/Downloads/ValidationGtBlocksRaw(2).mat'
    noisydatamat = loadmat(noisyraw_path)
    gtdatamat = loadmat(gtraw_path)
    noisy_rawdata = noisydatamat['ValidationNoisyBlocksRaw']
    gt_rawdata = gtdatamat['ValidationGtBlocksRaw']
    datarange=1023
    ddof=1

    batch, block, h, w = noisy_rawdata.shape

    sample_one_noisy = noisy_rawdata[0, :, :, :]
    sample_one_gt = gt_rawdata[0, :, :, :]
    #print(sample_one_gt.shape)
    noisy_bayers = split_bayer(sample_one_noisy)
    gt_bayers = split_bayer(sample_one_gt)

    for i in range(4):
        noisy_bayer = noisy_bayers[i]
        print(noisy_bayer.shape)
        gt_bayer = gt_bayers[i]
        z = compute_ab(noisy_bayer, gt_bayer, datarange, ddof, plot=True)
        a, b = z
        print(a, b)
'''

if __name__ == '__main__':
    '''
    noisy_folder = '/opt/data/private/Rawdata/SIDD_val_raw/noisy'
    denoised_folder ='/opt/data/private/ab_tuning/results/CResMDNet_Linearscale/train_abtuning_CResMD_Linearscale_continue.sh_bsd400_nf64_nb32_lr0.00025_step40000_batch16_patch64_LQGT_ab_dataset_a3e-05-0.05_b1e-07_highratio5_lossl1_Glo-Linear_scale-2161_Loc-Linear_scale-264/siddval/round1'
    get_ab_keys(noisy_folder, denoised_folder, indexnum=40)
    '''

    '''`+
    noisy_folder = '/opt/data/private/Rawdata/SIDD_val_raw/noisy'
    denoised_folder ='/opt/data/private/ab_tuning/results/CResMDNet/train_abtuning_CResMD_rawdata.sh_sidd-smallraw_nf64_nb32_lr0.0005_step40000_batch16_patch128_Train_rawpatch_a3e-05-0.05_b0_highratio0.01_lossl1/siddval_denoisedbsd/round1'
    get_ab_keys(noisy_folder, denoised_folder, indexnum=40)
    '''
    
    
    noisy_folder = '/opt/data/private/Rawdata/SIDD_val_raw/noisy'
    #denoised_folder ='/opt/data/private/ab_tuning/results/CResMDNet_Linearscale/train_abtuning_CResMD_Linearscale_continue.sh_bsd400_nf64_nb32_lr0.00025_step40000_batch16_patch64_LQGT_ab_dataset_a3e-05-0.05_b1e-07_highratio5_lossl1_Glo-Linear_scale-2161_Loc-Linear_scale-264/siddval/round1'
    gt_folder = '/opt/data/private/Rawdata/SIDD_val_raw/gt'
    minsamplenum = 5
    get_ab_keys_cookd_rmoutlier(noisy_folder, gt_folder, indexnum=40, minsamplenum=minsamplenum)










