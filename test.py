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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.patches as patches
import matplotlib
import scipy.io as io
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import pyplot as plt
import matplotlib.patches as Patches
import PIL.Image as img

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
    save_dir = join(project_root, '%s' % 'results')
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    # Specify network
    sep_net = MEBCRN()  
    criterion = NRMSELoss()
    
    if cuda:
        sep_net = sep_net.cuda()
        criterion.cuda()
    cudnn.benchmark = True
    
    mat1 = h5py.File('./data/test.mat')
    test = np.array(np.transpose(mat1['test'])) #[32, 8, 160, 224], "[num_slice, num_echo, width, height]", complex data
    
    mat2 = h5py.File('./data/testGroundtruth.mat')
    testGroundtruth = np.array(np.transpose(mat2['testGroundtruth'])) #[32, 2, 160, 224], "[num_slice, water/fat, width, height]", complex data
    del mat1, mat2

    sep_net.load_state_dict(torch.load('./data/MEBCRN_10rb_8echo_epoch_120.pth')) # Load the weights of the model obtained by training
    
    mean1 = [-161.1807071382206, 1361.1654357147088]    # Calculated by train.Py 
    std1 = [14395.118747473216, 14419.118940669792]
    mean2 = [-2337.518837319076, 1575.591773519404]
    std2 = [14583.30640687689, 14529.459839751646]

    test = to_tensor_format(test)
    test[:, 0, :, :, :] = (test[:, 0, :, :, :] - mean1[0]) / std1[0]
    test[:, 1, :, :, :] = (test[:, 1, :, :, :] - mean1[1]) / std1[1]
    test = Variable(torch.from_numpy(test).type(Tensor))

    testGroundtruth = to_tensor_format(testGroundtruth)
    testGroundtruth[:, 0, :, :, :] = (testGroundtruth[:, 0, :, :, :] - mean2[0]) / std2[0]
    testGroundtruth[:, 1, :, :, :] = (testGroundtruth[:, 1, :, :, :] - mean2[1]) / std2[1]
    testGroundtruth = np.transpose(testGroundtruth, (0, 4, 1, 2, 3)).reshape(32, 4, 160, 224)
    testGroundtruth = Variable(torch.from_numpy(testGroundtruth).type(Tensor))
    
    # test
    vis, vib, vid = [], [], []
    test_err, test_batches, test_ssim_water, test_ssim_fat, test_psnr_water, test_psnr_fat, t12 = 0, 0, 0, 0, 0, 0, 0
    for im in iterate_minibatch(test, batch_size, shuffle=False):
        t1 = time.time()
        pred = sep_net(im, test=True)  # pred(batch-size,4,160,224)
        del im
        gnd_in = gt_input(testGroundtruth, test_batches, batch_size)  # gnd(batch-size,2,160,224)
        test_err += float(criterion(pred, gnd_in))

        gnd_in = gnd_in.view(batch_size, 2, 2, Nx, Ny)  # (batch-size, 4, 160, 224)-->(batch-size, 2, 2, 160, 224)
        pred = pred.view(batch_size, 2, 2, Nx, Ny)
        gndb = from_tensor_format(gnd_in.data.cpu().numpy())
        predb = from_tensor_format(pred.data.cpu().numpy())
        
        for gnd_u, pred_u in zip(gndb, predb):
            test_psnr_fat,test_psnr_water,test_ssim_water,test_ssim_fat = 0,0,0,0
            water_pred, fat_pred, water_gnd, fat_gnd = pred_u[0, :, :], pred_u[1, :, :], gnd_u[0, :,:], gnd_u[1, :,:]
            
            test_psnr_water = complex_psnr(water_gnd, water_pred, peak='max')  
            test_psnr_fat = complex_psnr(fat_gnd, fat_pred, peak='max')  
            vid.append((test_psnr_fat + test_psnr_water) / 2)
            
            test_ssim_water = compare_ssim(abs(water_pred), abs(water_gnd), multichannel=False)
            test_ssim_fat = compare_ssim(abs(fat_pred), abs(fat_gnd), multichannel=False)
            vib.append((test_ssim_fat + test_ssim_water) / 2)
            
            vis.append((gnd_u[0, :,:], gnd_u[1, :,:], pred_u[0, :, :], pred_u[1, :, :]))
        
        test_batches += 1
        t11 = time.time()
        t12 += (t11 - t1)
        del gnd_u, pred_u, pred, gnd_in, gndb, predb
        
    
    t = t12 / (test_batches * batch_size)
    print(" separation time of each slice: {:.6f}s".format(t))
    print(" PSNR:  ", vid)
    print(" SSIM:  ", vib)
    print(" test_PSNR:      \t{:.6f}".format(np.sum(vid)/np.size(vid)))
    print(" test_SSIM:      \t{:.6f}".format(np.sum(vib)/np.size(vib)))
    print(" separation time of each slice: {:.6f}s".format(t))
   
    i = 0
    for gnd1_i, gnd2_i, pred1_i, pred2_i in vis:
        im1 = np.concatenate([np.abs(gnd1_i), np.abs(pred1_i), np.abs(gnd2_i), np.abs(pred2_i)], 1)
        plt.imsave(join(save_dir, 'im{}_{}.png'.format(str(1), i)), im1, cmap='gray')
        
        x1, x2, y1, y2 = 145, 170, 60, 80
        fig,ax = plt.subplots(dpi=400)
        ax.imshow(np.abs(gnd1_i), cmap="gray")
        ax.add_patch(patches.Rectangle((x1, y1), 25, 20, linewidth=1.5, edgecolor='red', fill=False))
        ax.axis("off")
        axins = zoomed_inset_axes(ax, 2, loc=2)  
        axins.imshow(np.abs(gnd1_i[y1:y2, x1:x2]), cmap="gray")
        axins.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./results/im%s_%d_%s.eps' % (str(1), 2, "Wgt"), bbox_inches='tight', dpi=400, pad_inches=0)
        del ax,fig,axins
        
        fig, ax1 = plt.subplots(dpi=400)
        ax1.imshow(np.abs(pred1_i), cmap="gray")
        ax1.add_patch(patches.Rectangle((x1, y1), 25, 20, linewidth=1.5, edgecolor='red', fill=False))
        ax1.axis("off")
        axins1 = zoomed_inset_axes(ax1, 2, loc=2)  
        axins1.imshow(np.abs(pred1_i[y1:y2, x1:x2]), cmap="gray")
        axins1.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./results/im%s_%d_%s.eps' % (str(1), 2, "Wpre"), bbox_inches='tight', dpi=400,pad_inches=0)
        del ax1, fig,axins1
        
        fig, ax2 = plt.subplots(dpi=400)
        ax2.imshow(np.abs(gnd2_i), cmap="gray")
        ax2.add_patch(patches.Rectangle((x1, y1), 25, 20, linewidth=1.5, edgecolor='red', fill=False))
        ax2.axis("off")
        axins2 = zoomed_inset_axes(ax2, 2, loc=2)  
        axins2.imshow(np.abs(gnd2_i[y1:y2, x1:x2]), cmap="gray")
        axins2.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./results/im%s_%d_%s.eps' % (str(1), 2, "Fgt"), bbox_inches='tight', dpi=400,pad_inches=0)
        del ax2, fig,axins2
        
        fig, ax3 = plt.subplots(dpi=400)
        ax3.imshow(np.abs(pred2_i), cmap="gray")
        ax3.add_patch(patches.Rectangle((x1, y1), 25, 20, linewidth=1.5, edgecolor='red', fill=False))
        ax3.axis("off")
        axins3 = zoomed_inset_axes(ax3, 2, loc=2)  
        axins3.imshow(np.abs(pred2_i[y1:y2, x1:x2]), cmap="gray")
        axins3.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./results/im%s_%d_%s.eps' % (str(1), 2, "Fpre"), bbox_inches='tight', dpi=400,pad_inches=0)
        
        fig = plt.figure(dpi=200)
        cnorm = matplotlib.colors.Normalize(vmin=0, vmax=0.15)
        m = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=matplotlib.cm.jet)
        c = np.abs(np.abs(gnd1_i) - np.abs(pred1_i))/np.max(np.max(np.abs(gnd1_i)))
        d = np.abs(np.abs(gnd2_i) - np.abs(pred2_i))/np.max(np.max(np.abs(gnd2_i)))
        m.set_array(c)
        plt.imshow(c, norm=cnorm, cmap="jet")
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./results/%s_%d_%s.png' % ('bcr', i, "w"), bbox_inches='tight',dpi=400,pad_inches=0)

        fig = plt.figure(dpi=200)
        m.set_array(d)
        plt.imshow(d, norm=cnorm, cmap="jet")
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./results/%s_%d_%s.png' % ('bcr', i, "f"),bbox_inches='tight', dpi=400,pad_inches=0)
        i += 1
    del vis
