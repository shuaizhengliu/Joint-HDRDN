import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
#import cv2
from scipy.io import savemat
import argparse
import sys
import os
import pdb
import pickle
import dataset as datalib
from dataset import DatasetNpy_Mix
from tools import load_weights
from utils import mu_tonemap, batch_rawPSNR, batch_rawSSIM, pack_raw_numpy, test_single_img, np_mu_tonemap, rebuild
import warnings
#warnings.filterwarnings("ignore")
import datetime
from PIL import Image
import importlib
from thop import profile
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=1, help='whether using cuda')
parser.add_argument('--gpus', type=int, default=1, help='folder to sr results')
parser.add_argument('--workers', type=int, default=0, help='number of threads to prepare data.')
parser.add_argument('--checkpoint', type=str, default='.', help='the checkpoint file')
parser.add_argument('--train_opt_file', type=str, help='the file saving training options')
parser.add_argument('--testfolder', type=str, default='',nargs='+', help='folder of test img')
parser.add_argument('--ori_testfolder', type=str, default='',nargs='+', help='folder of test img')
parser.add_argument('--model_module', type=str, help='the module including the model class for test')
parser.add_argument('--model_type', type=str, default='', help='the model class for test')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--result_dir', type=str, help ='the dir for saving result')
# Setup Params
opt = parser.parse_args()
use_cuda = opt.use_cuda
device = torch.device('cuda') if use_cuda else torch.device('gpu')

# load training options
with open(opt.train_opt_file, 'rb') as f:
    train_opt = pickle.load(f)


result_dir = opt.result_dir
patch_raw_dir = os.path.join(result_dir, 'patch_raw')
integral_raw_dir = os.path.join(result_dir, 'integral_raw')
if not os.path.exists(patch_raw_dir):
    os.makedirs(patch_raw_dir)
if not os.path.exists(integral_raw_dir):
    os.makedirs(integral_raw_dir)


# Logs
log_f = open(os.path.join(result_dir, 'logs.txt'), 'a') 

# Prepare Dataset
print('Preparing dataset ...')
print(opt.testfolder)
dataset_test = DatasetNpy_Mix(data_folder_list=opt.testfolder, patch_size=None, training=False)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=opt.workers, drop_last=False)

# Buliding network
print('Build Network ...')
modelmodule= importlib.import_module('models.'+ opt.model_module)
model_type = getattr(modelmodule, opt.model_type)

if train_opt.model_type in ['AHDR', 'DAHDR']:
    net = model_type()
else:
    net = model_type(train_opt)


# compute the total parameters of models
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))

model_file = opt.checkpoint
net = load_weights(net, model_file, opt.gpus, init_method='kaiming', scale=0.1)
net = net.to(device)
net.eval()

psnr_mu, psnr_l, ssim_mu, ssim_l = [], [], [], []
for batch, data in enumerate(test_loader):

    with torch.no_grad():
        im1, im2, im3, ref_hdr, save_name = data['sht'], data['mid'], data['lng'], data['hdr'], data['save_name']
        im1 = im1.to(device)
        im2 = im2.to(device)
        im3 = im3.to(device)
        ref_hdr = ref_hdr.to(device)
        generate_hdr = net(im1, im2, im3)
        generate_hdr = torch.clamp(generate_hdr, min=0.0, max=1.0).cpu()

    patch_num = len(save_name)
    for j in range(patch_num):
        savename = save_name[j]
        cur_hdr = generate_hdr[j]
        savepath = os.path.join(patch_raw_dir, savename+'.npy')
        np.save(savepath, cur_hdr)
        

#begin to rebuild result by assembling patch
print('begin to rebuild result')
psnr_mu = []
psnr_l = []
ssim_mu = []
ssim_l = []
for cur_test_folder in opt.ori_testfolder:
    test_list = os.listdir(cur_test_folder)
    for scene in test_list:
        scene_name = os.path.splitext(scene)[0]
        print('scene_name', scene_name)
        scene_file = os.path.join(cur_test_folder, scene)
        scene_gt = np.load(scene_file)['hdr']
        scene_gt_mu = np_mu_tonemap(scene_gt)
        c, h, w = scene_gt.shape
        patch_npy_list = glob.glob(os.path.join(patch_raw_dir, scene_name+'*.npy'))
        scene_result, tmp_h, tmp_w = rebuild(patch_npy_list, h, w, c) 
        scene_result_mu = np_mu_tonemap(scene_result)

        # compute evaluataion metric for each file
        scene_gt_mu = scene_gt_mu[:, :tmp_h, :tmp_w]
        scene_gt = scene_gt[:, :tmp_h, :tmp_w]
        psnr_mu_ = compare_psnr(scene_gt_mu, scene_result_mu, data_range=1)
        psnr_l_ = compare_psnr(scene_gt, scene_result, data_range=1)
        ssim_mu_ = structural_similarity(scene_gt_mu, scene_result_mu, data_range=1, channel_axis=0)
        ssim_l_ = structural_similarity(scene_gt, scene_result, data_range=1, channel_axis=0)
        psnr_mu.append(psnr_mu_)
        psnr_l.append(psnr_l_)
        ssim_mu.append(ssim_mu_)
        ssim_l.append(ssim_l_)
        print('psnr_mu of %s is %.3f' % (scene_name, psnr_mu_))
        print('psnr_l of %s is %.3f' % (scene_name, psnr_l_))
        print('ssim_mu of %s is %.3f' % (scene_name, ssim_mu_))
        print('ssim_l of %s is %.3f' % (scene_name, ssim_l_))
        log_f.write('psnr_mu of %s is %.3f' % (scene_name, psnr_mu_) + '\n')
        log_f.write('psnr_l of %s is %.3f' % (scene_name, psnr_l_) + '\n')
        log_f.write('ssim_mu of %s is %.3f' % (scene_name, ssim_mu_) + '\n')
        log_f.write('ssim_l of %s is %.3f' % (scene_name, ssim_l_) + '\n')           
        
        #transfer to raw to save
        result_raw = pack_raw_numpy(scene_result, 'BGGR')
        savename = scene_name+'.mat'
        savemat(os.path.join(integral_raw_dir, savename), {'rawhdr': result_raw})

average_psnr_mu = np.mean(psnr_mu)
average_psnr_l = np.mean(psnr_l)
average_ssim_mu = np.mean(ssim_mu)
average_ssim_l = np.mean(ssim_l)

print('the average psnr_mu is %.5f' %(average_psnr_mu))
print('the average psnr_l is %.5f' %(average_psnr_l))
print('the average ssim_mu is %.5f' %(average_ssim_mu))
print('the average ssim_l is %.5f' %(average_ssim_l))
log_f.write('the average psnr_mu is %.5f' %(average_psnr_mu)  + '\n')
log_f.write('the average psnr_l is %.5f' %(average_psnr_l)+ '\n')
log_f.write('the average ssim_mu is %.5f' %(average_ssim_mu)+ '\n')
log_f.write('the average ssim_l is %.5f' %(average_ssim_l)+ '\n')
