import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce
# from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch import nn

from torchvision.utils import make_grid, save_image

from os.path import join
import random
import numpy as np

class DAE(nn.Module):

    def __init__(self, nz, corr, sigma, dataset, fSize=2, multimodalZ=False):  #sigma is the corruption level
        super(DAE, self).__init__()
        #define layers here

        self.fSize = fSize
        self.nz = nz
        self.sigma = sigma
        self.multimodalZ = multimodalZ
        self.corr = corr
        self.dataset = dataset
    
        # self.imSize = imSize
        # inSize = int(imSize / ( 2 ** 4))
        # self.inSize = inSize

        if multimodalZ:
            NZ = 2
        else:
            NZ = nz
        
        if self.dataset == 'PAMAP2':
            self.enc1 = nn.Conv2d(1, fSize, 5, stride=2, padding=2)
            self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
            self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
            self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
            self.enc5 = nn.Linear(11264, NZ)

            self.dec1 = nn.Linear(NZ, 11264)
            self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
            self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 2, stride=2, padding=1, output_padding=1)
            self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
            self.dec5 = nn.ConvTranspose2d(fSize, 1, 2, stride=2, padding=1, output_padding=1)
            
        elif self.dataset == 'OPPO':
            self.enc1 = nn.Conv2d(1, fSize, 5, stride=2, padding=2)
            self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
            self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
            self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
            self.enc5 = nn.Linear(8192, NZ)

            self.dec1 = nn.Linear(NZ, 8192)
            self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 2, stride=2, padding=1, output_padding=1)
            self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 2, stride=2, padding=[1,0], output_padding=[1,0])
            self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 2, stride=2, padding=[1,0], output_padding=[1,0])
            self.dec5 = nn.ConvTranspose2d(fSize, 1, 2, stride=2, padding=[1,0], output_padding=[1,0])            
        
        self.relu = nn.ReLU()
        self.useCUDA = torch.cuda.is_available()

    def norm_prior(self, noSamples=25):
        z = torch.randn(noSamples, self.nz)
        return z

    def multi_prior(self, noSamples=25, mode=None):
        #make a 2D sqrt(nz)-by-sqrt(nz) grid of gaussians
        num = np.sqrt(self.nz) #no of modes in x and y
        STD = 1.0
        modes = np.arange(-num,num)
        p = np.random.uniform(0, num,(noSamples*2))

        if mode is None:
            mu = modes[np.floor(2 * p).astype(int)]
        else:
            mu = modes[np.ones((noSamples, 2), dtype=int) * int(mode)]

        z = torch.Tensor(mu).view(-1,2) + STD * torch.randn(noSamples, 2)
        return z

    def encode(self, x):
        #define the encoder here return mu(x) and sigma(x)
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        x = self.relu(self.enc4(x))
        x = x.view(x.size(0), -1)
        x = self.enc5(x) 

        return x

    def corrupt(self, x):
        if self.corr == 'Gaussian':
            noise = self.sigma[0] * torch.randn(x.size()).type_as(x)
            return x + noise
        elif self.corr == 'ZeroMask':
            num_zeros = int(torch.numel(x) * self.sigma[0])
            mask = torch.ones(torch.numel(x), device=x.device)
            mask[:num_zeros] = 0
            mask = mask[torch.randperm(mask.shape[0])]
            mask = mask.reshape(x.shape)
            return torch.mul(x, mask)
        elif self.corr == 'ConsecutiveZeros':
            # print(x.shape)
            # time * batch_size * feature
            # lambdas reuse the sigma variable, unpack
            lambda_corr = self.sigma[0] # lambda for missing data period
            lambda_norm = self.sigma[1] # lambda for normal data periodß
            
            mask = np.ones(x.shape)
            for sample_id in range(mask.shape[0]):
                for ch_id in range(mask.shape[2]):  
                    ptr = 0
                    is_corrupted = False
                    while ptr < mask.shape[3]:
                        if is_corrupted:
                            corr_duration = int(np.random.exponential(scale=lambda_corr))
                            #  mask[ptr:min(mask.shape[0], ptr + corr_duration), sample_id, ch_id] = 0
                            mask[sample_id, 0 ,ch_id, ptr:min(mask.shape[3], ptr + corr_duration)] = 0
                            ptr = min(mask.shape[3], ptr + corr_duration)
                            is_corrupted = False
                        else:
                            norm_duration = int(np.random.exponential(scale=lambda_norm))
                            ptr = min(mask.shape[3], ptr + norm_duration)
                            is_corrupted = True
            x = torch.mul(x, torch.from_numpy(mask).type_as(x))
            return x
        elif self.corr == 'Both':
            noise = self.sigma[0] * torch.randn(x.size()).type_as(x)
            x = x + noise
            # print(x.shape)
            # time * batch_size * feature
            # lambdas reuse the sigma variable, unpack
            lambda_corr = self.sigma[1] # lambda for missing data period
            lambda_norm = self.sigma[2] # lambda for normal data periodß

            mask = np.ones(x.shape)
            for sample_id in range(mask.shape[0]):
                for ch_id in range(mask.shape[2]):  
                    ptr = 0
                    is_corrupted = False
                    while ptr < mask.shape[3]:
                        if is_corrupted:
                            corr_duration = int(np.random.exponential(scale=lambda_corr))
                            #  mask[ptr:min(mask.shape[0], ptr + corr_duration), sample_id, ch_id] = 0
                            mask[sample_id, 0 ,ch_id, ptr:min(mask.shape[3], ptr + corr_duration)] = 0
                            ptr = min(mask.shape[3], ptr + corr_duration)
                            is_corrupted = False
                        else:
                            norm_duration = int(np.random.exponential(scale=lambda_norm))
                            ptr = min(mask.shape[3], ptr + norm_duration)
                            is_corrupted = True
            x = torch.mul(x, torch.from_numpy(mask).type_as(x))
            return x
        else:
            raise NotImplementedError

    
        


    def sample_z(self, noSamples=25, mode=None):
        if not self.multimodalZ:
            z = self.norm_prior(noSamples=noSamples)
        else:
            z = self.multi_prior(noSamples=noSamples, mode=mode)
        if self.useCUDA:
            return z.cuda()
        else:
            return z

    def decode(self, z):
        #define the decoder here
        # print('z')
        z = self.relu(self.dec1(z))
        # z = z.view(z.size(0), -1, self.inSize, self.inSize)
        if self.dataset == 'PAMAP2':
            z = z.view(z.size(0), -1, 2, 11)
        elif self.dataset == 'OPPO':
            z = z.view(z.size(0), -1, 8, 2)

        
        z = self.relu(self.dec2(z))
        z = self.relu(self.dec3(z))
        z = self.relu(self.dec4(z))
        z = torch.sigmoid(self.dec5(z))

        return z

    def forward(self, x):
        # the outputs needed for training
        # x_corr = self.corrupt(x, rows =4, noise_perc=noise_perc)
        x_corr = self.corrupt(x)
        z = self.encode(x_corr)
        return z, self.decode(z)

    def rec_loss(self, rec_x, x, loss='BCE'):
        if loss == 'BCE':
            return torch.mean(bce(rec_x, x, size_average=True))  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
        elif loss == 'MSE':
            return torch.mean(F.mse_loss(rec_x, x, size_average=True))
        else:
            print('unknown loss:'+loss)

    def save_params(self, exDir):
        print('saving params...')
        torch.save(self.state_dict(), join(exDir, 'dae_params'))

    def load_params(self, exDir):
        print('loading params...')
        self.load_state_dict(torch.load(join(exDir, 'dae_params')))

    def sample_x(self, M, exDir, z=None):
        if z == None:
            z = self.sample_z(noSamples=25)
        if not self.multimodalZ:
            x_i = self.decode(z)
            save_image(x_i.data, join(exDir, 'samples0.png'))
            for i in range(M):
                z_i, x_i = self.forward(x_i) #corruption already in there!
                save_image(x_i.data, join(exDir, 'samples'+str(i+1)+'.png'))
        else:
            #show samples from a few modes
            maxModes = min(self.nz, 5)  #show at most 5 modes
            for mode in range(maxModes):
                z = self.sample_z(noSamples=25, mode=mode)
                x_i = self.decode(z)
                save_image(x_i.data, join(exDir, 'mode'+str(mode)+'_samples.png'))
                for i in range(M):
                    z_i, x_i = self.forward(x_i) #corruption already in there!
                    save_image(x_i.data, join(exDir, 'mode'+str(mode)+'_samples'+str(i+1)+'.png'))