import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import argparse

"""
nhdrrnet -> based on current UNet Structure
"""

class Encoder(nn.Module):
    """
    the similar structure as the ECCV paper
    """
    def __init__(self, in_channels, nFeat):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, nFeat, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(nFeat, nFeat*2, kernel_size=5, stride=2, padding=2)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(nFeat*2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn(self.conv2(self.relu(x1)))
        return x1, x2

class Merge(nn.Module):
    """
    merge the outputs
    """
    def __init__(self, in_channels, out_channels, num_residual=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        triple_layers = []
        for _ in range(num_residual):
            triple_layers.append(nn.LeakyReLU())
            triple_layers.append(TriplePassBlocks(out_channels))
        self.triple_blocks = nn.Sequential(*triple_layers)
        # get the non_local layer
        self.non_local = global_non_local()
    
    def forward(self, x):
        x1 = self.bn(self.conv(self.relu(x)))
        x2 = self.triple_blocks(x1)
        x3 = self.non_local(x1, (x2.shape[2], x2.shape[3]))
        return x2, x3

# define the global non-local
class global_non_local(nn.Module):
    def __init__(self):
        super().__init__()
        # define the 
        self.theta_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.phi_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.g_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.theta_phi_g_conv = nn.Conv2d(128, 256, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x, size):
        x = self.avgpool(x)
        b, c, h, w = x.shape
        theta = self.theta_conv(x).reshape(b, c//2, h * w).permute(0, 2, 1).contiguous()
        phi = self.phi_conv(x).reshape(b, c//2, h * w)
        g = self.g_conv(x).reshape(b, c//2, h * w).permute(0, 2, 1).contiguous()
        # theta
        theta_phi = F.softmax(torch.matmul(theta, phi),dim=-1)
        theta_phi_g = torch.matmul(theta_phi, g)
        theta_phi_g = theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, c//2, h, w)
        # get g
        theta_phi_g = self.theta_phi_g_conv(theta_phi_g)
        # get output
        output = theta_phi_g + x
        output = F.interpolate(output, size=size)
        return output

class TriplePassBlocks(nn.Module):
    def __init__(self, nFeat):
        super().__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nFeat, nFeat, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(3*nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        # remove the BN firstly
        #self.bn = nn.BatchNorm2d(nFeat)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # get three intermediate
        x_1 = self.relu(self.conv1(x))
        x_2 = self.relu(self.conv2(x))
        x_3 = self.relu(self.conv3(x))
        x_ = torch.cat([x_1, x_2, x_3], dim=1)
        x_ = self.conv4(x_)
        # then residual connection
        return x + x_

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # deconv
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.bn(self.deconv1(self.relu(x)))
        return x

class NHDRRNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_channel = args.n_channel
        self.encoder1 = Encoder(n_channel, 64)
        self.encoder2 = Encoder(n_channel, 64)
        self.encoder3 = Encoder(n_channel, 64)
        # define the merge block
        self.merge = Merge(64*2*3, 64*4)
        # define the decoder
        self.decoder1 = Decoder(64*4*2, 64*2)
        self.decoder2 = Decoder(64*4*2, 64)
        self.decoder3 = Decoder(64*4*1, 64)
        # conv
        self.conv_hdr = nn.Conv2d(64, 4, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        # encoder
        x1_1, x1_2 = self.encoder1(x1)
        x2_1, x2_2 = self.encoder2(x2)
        x3_1, x3_2 = self.encoder3(x3)
        # merge
        x_ = torch.cat([x1_2, x2_2, x3_2], dim=1)
        # send into the merger
        xm_1, xm_2 = self.merge(x_)
        # padding
        d_0 = torch.cat([xm_2, xm_1], dim=1)
        # 64 * 64
        d_1 = self.decoder1(d_0)
        # 128 * 128
        d_1 = torch.cat([d_1, x1_2, x2_2, x3_2], dim=1)
        d_2 = self.decoder2(d_1)
        # 256 * 256
        d_2 = torch.cat([d_2, x1_1, x2_1, x3_1], dim=1)
        d_3 = self.decoder3(d_2)
        # output
        out = self.conv_hdr(self.relu(d_3))
        # output
        return torch.sigmoid(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-channel', type=int, default=6, help='the number of input channels')
    args = parser.parse_args()
    # define the wavelet
    net = NHDRRNet(args)
    net.cuda()
    inputs = np.ones((1, 6, 256, 256), dtype=np.float32)
    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        outputs = net(inputs, inputs, inputs)
    print(outputs.shape)