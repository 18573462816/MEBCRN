import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
from net_pytorch.dnn_io import to_tensor_format
from net_pytorch.dnn_io import from_tensor_format
import matplotlib.pyplot as plt
import matplotlib

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

class ECHOcell(nn.Module):
    """ Convolutional ECHO cell that evolves over both echoes and iterations
        input: 4d tensor, shape (batch_size, channel, width, height)
                       hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
                    iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
        output: 4d tensor, shape (batch_size, hidden_size, width, height)    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ECHOcell, self).__init__()
        
        self.hidden_size = hidden_size 
        self.kernel_size = kernel_size  
        self.input_size = input_size 
        
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, input, iteration, hidden):
        in_to_hid = self.i2h(input)   #input convolution layer
        hid_to_hid = self.h2h(hidden)     #bidirectional convolution layer
        ih_to_ih = self.ih2ih(iteration)      #iteration convolution layer
        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        
        del in_to_hid,hid_to_hid,ih_to_ih
        return hidden
        
def creat_hid(nb, hidden_size, nx, ny,j, test=False ):

    size_h = [nb, (hidden_size//8+j*2), nx, ny]  
    
    if test:
        with torch.no_grad():
            hid_init = Variable(torch.zeros(size_h)).cuda()
    else:
        hid_init = Variable(torch.zeros(size_h)).cuda()
    return hid_init
    
class MEBCunit(nn.Module):
    """ Multi-echo Bidirectional Convolutional Unit
          incomings:     input: 5d tensor, [input_image] with shape (num_echo, batch_size, channel, width, height)
                        output: 5d tensor, [hidden states from previous iteration] with shape (n_echo, n_batch, hidden_size, width, height)
                          test: True if in test mode, False if in train mode
          output: 5d tensor, shape (n_echo, n_batch, hidden_size, width, height)    """
          
    def __init__(self, input_size, hidden_size, kernel_size,nr):
        super(MEBCunit, self).__init__()
        
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.nr = nr
        
        self.MBCNunit10 = ECHOcell(self.input_size, (self.hidden_size)//8, self.kernel_size)
        self.MBCNunit20 = ECHOcell(self.input_size, (self.hidden_size)//8+2, self.kernel_size)
        self.MBCNunit30 = ECHOcell(self.input_size, (self.hidden_size)//8+4, self.kernel_size)
        self.MBCNunit40 = ECHOcell(self.input_size, (self.hidden_size)//8+6, self.kernel_size)
        
        self.MBCNunit11 = ECHOcell(self.input_size, (self.hidden_size)//8, self.kernel_size)
        self.MBCNunit21 = ECHOcell(self.input_size, (self.hidden_size)//8+2, self.kernel_size)
        self.MBCNunit31 = ECHOcell(self.input_size, (self.hidden_size)//8+4, self.kernel_size)
        self.MBCNunit41 = ECHOcell(self.input_size, (self.hidden_size)//8+6, self.kernel_size)
        
    def forward(self, input, test=False):
    
        ne, nb, nc, nx, ny = input.shape 
        size_o = [ne,nb, self.hidden_size//8, nx, ny
        
        if test:
            with torch.no_grad():
                iteration = Variable(torch.zeros(size_o)).cuda()
        else:
            iteration = Variable(torch.zeros(size_o)).cuda()
            
        for j in range(self.nr):#0-4
            output_f = []
            output_b = []
            # forward
            for i in range(ne):#0-7
                if j == 0:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit10(input[i], iteration[i], hidden)
                if j == 1:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit20(input[i], iteration[i], hidden)
                if j == 2:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit30(input[i], iteration[i], hidden)
                if j == 3:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit40(input[i], iteration[i], hidden)
                output_f.append(hidden)
            output_f = torch.cat(output_f) 
            
            # backward
            for i in range(ne):
                if j == 0:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit11(input[ne - i - 1], iteration[ne - i - 1], hidden)
                if j == 1:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit21(input[ne - i - 1], iteration[ne - i - 1], hidden)
                if j == 2:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit31(input[ne - i - 1], iteration[ne - i - 1], hidden)
                if j == 3:
                    hidden = creat_hid(nb, self.hidden_size, nx, ny, j, test=False)
                    hidden = self.MBCNunit41(input[ne - i - 1], iteration[ne - i - 1], hidden)
                output_b.append(hidden)
            output_b = torch.cat(output_b[::-1])
            
            output = output_f + output_b   
            output = output.view(ne, nb, (self.hidden_size//8+j*2), nx, ny)
            iteration = torch.cat((output,input), dim=2) 
            
            del output_f, output_b, output
        del hidden,input
        
        return iteration

class MEBCRN(nn.Module):
    """  Model for water/fat separation using Convolutional Neural Networks
    incomings: three 5d tensors, [input_image] with shape (batch_size, 2, width, height, n_echo)
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, W/F) """
    
    def __init__(self, n_ch=2, nf=128, ks=3, nc=10, nr=4):
        """ :param n_ch: number of channels
            :param nf: number of filters
            :param ks: kernel size
            :param nr: number of MEBCunit iterations in MEBCN Block
            :param nc: number of RCA_Block iterations"""
        super(MEBCRN, self).__init__()
        
        self.nc = nc
        self.nr = nr
        self.nf = nf
        self.ks = ks
        
        self.MEBCN = MEBCunit(n_ch, 64,  ks, nr) 
        
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)  #conv 1 of the first RB block
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)  #conv 2 of the first RB block
        self.conv1_m = nn.Conv2d(nf, nf, 1, padding=0)   #conv 1_m of the MLFF
        
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv4_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv4_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv4_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv5_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv5_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv5_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv6_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv6_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv6_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv7_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv7_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv7_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv8_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv8_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv8_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv9_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv9_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv9_m = nn.Conv2d(nf, nf, 1, padding=0)
        
        self.conv10_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv10_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        
        self.conv11 = nn.Conv2d(nf, nf, ks, padding=ks // 2) 
        self.conv12 = nn.Conv2d(nf, 4,  ks, padding=ks // 2) 
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, test=False):
        """  x   - input in image domain, of shape (n, 2, nx, ny, n_echo)
            test - True: the model is in test mode, False: train mode """
        net = {}
        n_batch, n_ch, width, height, n_echo = x.size()
        x = x.permute(4, 0, 1, 2, 3)  
        x = x.contiguous()
        
        net['t_x0'] = self.MEBCN(x, test)
        net['t_x0'] = net['t_x0'].permute(1, 0, 2, 3, 4)
        net['t_x0'] = net['t_x0'].contiguous()
        del x
        
        net['t_x0'] = net['t_x0'].view(n_batch, 128, width, height)  
        
        for i in range(1, self.nc+1):
            if i == 1:
                net['t1_x1'] = self.conv1_x(net['t_x0'])
                net['t1_x1'] = self.relu(net['t1_x1'])
                
                net['t1_x2'] = self.conv1_h(net['t1_x1'])
                net['t1_x2'] = self.relu(net['t1_x2'])
                
                net['t1_x3'] = net['t_x0'] + net['t1_x2']
                
                net['t1_x4'] = self.conv1_m(net['t1_x3'])
                net['t1_x4'] = self.relu(net['t1_x4'])
                
                del net['t_x0']
            if i == 2:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv2_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t1_x3'], net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 3:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv3_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 4:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv4_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 5:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv5_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 6:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv6_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 7:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv7_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 8:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv8_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 9:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                net['t%d_x4' % i] = self.conv9_m(net['t%d_x3' % i])
                net['t%d_x4' % i] = self.relu(net['t%d_x4' % i])
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
            if i == 10:
                net['t%d_x1' % i] = self.conv2_x(net['t%d_x3' % (i - 1)])
                net['t%d_x1' % i] = self.relu(net['t%d_x1' % i])
                
                net['t%d_x2' % i] = self.conv2_h(net['t%d_x1' % i])
                net['t%d_x2' % i] = self.relu(net['t%d_x2' % i])                
               
                net['t%d_x3' % i] = net['t%d_x3' % (i - 1)] + net['t%d_x2' % i]
                
                del net['t%d_x3' % (i - 1)],net['t%d_x1' % (i)], net['t%d_x2' % (i)]
           
        net['t10_x4'] = net['t1_x4'] + net['t2_x4'] + net['t3_x4'] + net['t4_x4'] + net['t5_x4'] + net['t6_x4'] + net['t7_x4'] +net['t8_x4'] + net['t9_x4'] + net['t10_x3']
        
        net['t11'] = self.conv11(net['t10_x4'])
        net['t12'] = self.conv12(net['t11'])
        
        del net['t1_x4'],net['t2_x4'],net['t3_x4'],net['t4_x4'],net['t5_x4'],net['t6_x4'],net['t7_x4'],net['t8_x4'],net['t9_x4'],net['t10_x3'],net['t10_x4'],net['t11']
        
        return net['t12'] #(batch-size,4,160,224)
