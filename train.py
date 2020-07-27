from __future__ import print_function, division
from skimage.measure import compare_ssim, compare_nrmse, compare_psnr
import h5py
import os
import time
import copy
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from os.path import join
from scipy.io import loadmat

from utils.metric import complex_psnr
from net_pytorch.model_pytorch8echo_10RAs import *
from net_pytorch.dnn_io import to_tensor_format
from net_pytorch.dnn_io import from_tensor_format


def gt_input(gt, a, batch_size):
    gnd_in = gt[(a * batch_size):(a * batch_size + batch_size), :, :, :]
    return gnd_in

def iterate_minibatch(data, batch_size, shuffle=False):
    n = len(data)  
    if shuffle:
        data = np.random.permutation(data) 
    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]

class NRMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(NRMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self.eps) / torch.sqrt(torch.mean(y ** 2))
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['120'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['4'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1, default=['0.001'],
                        help='initial learning rate')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--savefig', action='store_true', default=True,
                        help='Save output images ')
    args = parser.parse_args()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Project config
    model_name = 'MEBCRN_10rb_8echo'
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny, echo, nc, save_every = 160, 224, 8, 2, 1
    save_fig = args.savefig

    # Configure directory info
    project_root = '.'
    save_dir = join(project_root, '%s' % model_name)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    # Specify network
    sep_net = MEBCRN()  
    
    criterion = NRMSELoss()
    
    if cuda:
        sep_net = sep_net.cuda()
        criterion.cuda()
    cudnn.benchmark = True
    
    mat1 = h5py.File('./data/train.mat')
    train = np.array(np.transpose(mat1['train'])) #[324, 8, 160, 224], "[num_slice, num_echo, width, height]", complex data
    
    mat2 = h5py.File('./data/trainGroundtruth.mat')
    trainGroundtruth = np.array(np.transpose(mat2['trainGroundtruth'])) #[324, 2, 160, 224], "[num_slice, water/fat, width, height]", complex data
    
    mat3 = h5py.File('./data/val.mat')
    val = np.array(np.transpose(mat3['val'])) #[28, 8, 160, 224], "[num_slice, num_echo, width, height]", complex data
    
    mat4 = h5py.File('./data/valGroundtruth.mat')
    valGroundtruth = np.array(np.transpose(mat4['valGroundtruth'])) #[28, 2, 160, 224], "[num_slice, water/fat, width, height]", complex data
    del mat1, mat2 ,mat3, mat4  

    j = 0
    for epoch in range(num_epoch):
        t_start = time.time()
        
        train_indices = np.random.permutation(train_indices)   # shuffle
        
        train_input = to_tensor_format(train[train_indices])   # torch.Size([324, 2, 160, 224, 8]), "[num_slice, real/imag, width, height, num_echo]", complex data--> 2 channels of real+imag data
        trainGroundtruth_input = to_tensor_format(trainGroundtruth[train_indices])   # torch.Size([324, 2, 160, 224, 2]), "[num_slice, real/imag, width, height, water/fat]", complex data--> 2 channels of real+imag data

        if epoch == 0:

            mean1, mean2, std1, std2 = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
            for ms in range(2):             # Calculating the mean and standard deviation of real and imaginary channels
                mean1[ms] += train_input[:, ms, :, :,:].mean()  
                std1[ms] += train_input[:, ms, :, :,:].std()  
                mean2[ms] += trainGroundtruth_input[:, ms, :, :,:].mean()  
                std2[ms] += trainGroundtruth_input[:, ms, :, :,:].std() 

            val = to_tensor_format(val)
            val[:, 0, :, :, :] = (val[:, 0, :, :, :] - mean1[0]) / std1[0]
            val[:, 1, :, :, :] = (val[:, 1, :, :, :] - mean1[1]) / std1[1]
            val = Variable(torch.from_numpy(val).type(Tensor))

            valGroundtruth = to_tensor_format(valGroundtruth)
            valGroundtruth[:, 0, :, :, :] = (valGroundtruth[:, 0, :, :, :] - mean2[0]) / std2[0]
            valGroundtruth[:, 1, :, :, :] = (valGroundtruth[:, 1, :, :, :] - mean2[1]) / std2[1]
            valGroundtruth = np.transpose(valGroundtruth, (0, 4, 1, 2, 3)).reshape(28, 4, 160, 224)
            valGroundtruth = Variable(torch.from_numpy(valGroundtruth).type(Tensor))

        train_input[:, 0, :, :, :] = (train_input[:, 0, :, :, :] - mean1[0]) / std1[0]
        train_input[:, 1, :, :, :] = (train_input[:, 1, :, :, :] - mean1[1]) / std1[1]
        train_input = Variable(torch.from_numpy(train_input).type(Tensor))  

        trainGroundtruth_input[:, 0, :, :, :] = (trainGroundtruth_input[:, 0, :, :, :] - mean2[0]) / std2[0]
        trainGroundtruth_input[:, 1, :, :, :] = (trainGroundtruth_input[:, 1, :, :, :] - mean2[1]) / std2[1]
        trainGroundtruth_input = np.transpose(trainGroundtruth_input, (0, 4, 1, 2, 3)).reshape(324, 4, 160, 224)
        trainGroundtruth_input = Variable(torch.from_numpy(trainGroundtruth_input).type(Tensor))
        
        j += 1
        optimizer = optim.Adam(sep_net.parameters(), lr=float(args.lr[0]) * (0.95 ** np.trunc((epoch + 1) / 5)),betas=(0.9, 0.999))
        lr = float(args.lr[0]) * (0.95 ** np.trunc((epoch + 1) / 5))
        
        # Training
        train_err, train_batches = 0, 0
        for im in iterate_minibatch(train_input, batch_size, shuffle=False):
            optimizer.zero_grad()
            spred = sep_net(im, test=False)  
            gnd_in = gt_input(trainGroundtruth_input, train_batches, batch_size) 
            loss = criterion(pred, gnd_in)
            loss.backward()
            optimizer.step()
            train_err += float(loss)
            train_batches += 1
            del pred, gnd_in, im
            
        train_err /= (train_batches * batch_size)
        del loss, train_input, trainGroundtruth_input
        
        if epoch > (100):
            name = '%s_epoch_%d.pth' % (model_name, epoch + 1)
            torch.save(sep_net.state_dict(), join(save_dir, name))
            print('model parameters saved at %s' % join(os.getcwd(), name))

        # val
        val_err, val_batches = 0, 0
        for im in iterate_minibatch(val, batch_size, shuffle=False):
            pred = sep_net(im, test=True)  
            gnd_in = gt_input(valGroundtruth, val_batches, batch_size)
            val_err += float(criterion(pred, gnd_in))

            val_batches += 1
            del im, pred, gnd_in

        val_err /= (val_batches * batch_size)
        t_end = time.time()
        
        print("")
        print(" Epoch {}/{}".format(epoch + 1, num_epoch))
        print(" Learning rate: \t{:.6f}".format(lr))
        print(" training loss: \t{:.6f}".format(train_err))
        print(" val loss:      \t{:.6f}".format(val_err))
        print(" time:          \t{:.6f}s".format(t_end - t_start))
        print("****************************************")
        print("")
