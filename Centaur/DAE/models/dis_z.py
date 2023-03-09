import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce
from os.path import join


class DIS_Z(nn.Module):

    '''
    Discriminate between z_real and z_fake vectors
    '''

    def __init__(self, nz, prior):  #cannot use torch.randn anymore cause wen called does not know nz
        super(DIS_Z, self).__init__()

        self.nz = nz
        self.prior = prior

        self.dis1 = nn.Linear(nz, 1000)
        self.dis2 = nn.Linear(1000, 1000)
        self.dis3 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def discriminate(self, z):
        z = self.relu(self.dis1(z))
        z = self.relu(self.dis2(z))
        z = torch.sigmoid(self.dis3(z))
        return z

    def forward(self, z):
        return self.discriminate(z)

    def dis_loss(self, z):
        zReal = self.prior(z.size(0)).type_as(z)
        pReal = self.discriminate(zReal)

        zFake = z.detach()  #detach so grad only goes thru dis
        pFake = self.discriminate(zFake)

        ones = torch.Tensor(pReal.size()).fill_(1).type_as(pReal)
        zeros = torch.Tensor(pFake.size()).fill_(0).type_as(pFake)

        return 0.5 * torch.mean(bce(pReal, ones) + bce(pFake, zeros))

    def gen_loss(self, z):
        # n.b. z is not detached so it will update the models it has passed thru
        pFake = self.discriminate(z)
        ones = torch.Tensor(pFake.size()).fill_(1).type_as(pFake)
        # ones = torch.ones_like(pFake)
        
        return torch.mean(bce(pFake, ones))

    def save_params(self, exDir):
        print('saving params...')
        torch.save(self.state_dict(), join(exDir, 'dis_z_params'))

    def load_params(self, exDir):
        print('loading params...')
        self.load_state_dict(torch.load(join(exDir, 'dis_z_params')))

    def plot_encz(self, exDir):  #TODO
        '''
        plot the encoded z samples
        '''
        print('TO DO')
