import torch
import numpy as np
import torch.nn as nn




##Attention block1 
def sa_conv_module(X, fweight, gweight, number_of_outputs):
    h = number_of_outputs
    f_transf_output = fweight(X)
    g_transf_output = gweight(X)
    M = f_transf_output*g_transf_output
    M = M/(h**0.5)
    M = M.mean(axis=1)
    M = M.mean(axis=-1)
    ## weights for attention.
    w = nn.Softmax(dim=-1)(M)
    return torch.einsum('bcntf,bnt->bcntf', X, w).sum(axis=3)

## Attention block2
def sa_temporal_module(X, fweight, gweight, number_of_outputs):
    h = number_of_outputs
    num_intervals = X.shape[1]
    X_extended = X.unsqueeze(1)
    f_transf_output = fweight(X_extended)
    g_transf_output = gweight(X_extended)
    M = f_transf_output*g_transf_output
    M = M/(h**(0.5))
    M = M.mean(axis=1)
    M = M.mean(axis=-1)
    w = nn.Softmax(dim=-1)(M)
    return torch.einsum('bnf,bn->bnf', X, w).sum(axis=1)


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.num_cells = 120
        self.sigma =0.2
        
        #self.sigma= [20, 50]
        #self.sigma =0.2
        #self.sigma= [20, 50]
        self.h = 6

        ## individual conv network1
        self.cn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= (1, 6*3), stride=(1,6)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        ## individual conv network2
        self.cn2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= (1, 6*3), stride=(1,6)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        ## individual conv network3
        self.cn3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= (1, 6*3), stride=(1,6)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
)
        ## merge convnet
        self.cn4 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 8), padding='same' ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, (1,6), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, (1,4),padding='same' ),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        ## GRU layers
        self.gru = nn.GRU(64*4, self.num_cells)
        self.gru2 = nn.GRU(self.num_cells, self.num_cells)

        ##attention weights
        self.convfweight = nn.Conv3d(64, self.h, 1)
        self.convgweight = nn.Conv3d(64, self.h, (1, 2, 1))

        self.tempfweight = nn.Conv2d(1, self.h, 1)
        self.tempgweight = nn.Conv2d(1, self.h, (20, 1))

        ## output layer
        #self.l1 = nn.Linear(120, 32)
        #self.act1= nn.ReLU()
        self.l2 = nn.Linear(120, 6)
        self.act2 = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.2)

    def corrupt1(self, x):
        noise = self.sigma * torch.randn(x.size()).type_as(x)
        return x + noise

    def corrupt2(self, x):
        # self.sigma =[40, 80]   # case 1
        # self.sigma =[50, 80]    case 2
        # self.sigma =[60, 80]    case 3

        lambda_corr = self.sigma[0] # lambda for missing data period
        lambda_norm = self.sigma[1] # lambda for normal data periodß
        # corrupted_x = copy.deepcopy(x)
        # failure_mat = np.random.uniform(size = x.shape) < failure_rate
        # num_failures = np.sum(failure_mat)
        # failure_durations = np.random.exponential(scale = duration_scale, size = num_failures).astype(int)
        mask = torch.ones_like(x)
        #print(x.shape)
        # failure_id = 0
        # sample_id is the batch size : 64
        # ch_id is the features: 27
        # mask.shape[0]: 171
        for sample_id in range(mask.shape[0]):
            for ch_id in range(mask.shape[2]):  
                ptr = 0
                is_corrupted = False
                while ptr < mask.shape[3]:
                    if is_corrupted:
                        corr_duration = int(np.random.exponential(scale=lambda_corr))
                        #  mask[ptr:min(mask.shape[0], ptr + corr_duration), sample_id, ch_id] = 0
                        mask[sample_id, [0] ,ch_id, ptr:min(mask.shape[3], ptr + corr_duration)] = 0
                        ptr = min(mask.shape[3], ptr + corr_duration)
                        is_corrupted = False
                    else:
                        norm_duration = int(np.random.exponential(scale=lambda_norm))
                        ptr = min(mask.shape[3], ptr + norm_duration)
                        is_corrupted = True
        return torch.mul(x, mask)   

    def corrupt3(self, x):
        # self.sigma =[0.1,40, 80]      #case 1
    # self.sigma =[0.2,50, 80]      case 2
    # self.sigma =[0.3,60, 80]      case 3


        noise = self.sigma[0] * torch.randn(x.size()).type_as(x)
        x = x + noise
         # print(x.shape)
         # time * batch_size * feature
         # lambdas reuse the sigma variable, unpack
        lambda_corr = self.sigma[1] # lambda for missing data period
        lambda_norm = self.sigma[2] # lambda for normal data periodß

        mask = torch.ones_like(x)
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
        x = torch.mul(x, mask)
        return x
    
    def forward(self, x1):
        ## unsqueeze(expand_dims) for concatenation.

        #mean = x1.mean()
        x1 = (x1 - x1.min()) / (x1.max() - x1.min())
        # std = x1.std()
        # x1 = (x1 - mean)/std
        #print(x1.min())
    
        #print('withour corr', norm)
        x1 = self.corrupt1(x1)
       # print(' corr', x1)
        
        # print((x1 == 0).sum())

        j1, j2 = torch.split(x1, x1.shape[-1]//2, dim=-1)
        o1 = self.cn1(j1).cuda().unsqueeze(3)
        o2 = self.cn2(j2).cuda().unsqueeze(3)
        # o3 = self.cn3(j3).unsqueeze(3)

#         ## concatenate along sensor dimension
        out = torch.cat((o1, o2), axis=3)
        out = self.drop(out)

#         ## first attention
        out = sa_conv_module(out, self.convfweight, self.convgweight, number_of_outputs=6)
#         return out

        ## merge conv
        out = self.cn4(out)
        # print(out.shape)

#         ## reshape for GRU
        out = out.view(-1, out.shape[2], 64*4)
        out, _ = self.gru(out)
        out, _ = self.gru2(out)

#         ##second attention
        out = sa_temporal_module(out, self.tempfweight, self.tempgweight, number_of_outputs=6)

#         ## output network.
       # out = self.act1(self.l1(out))
        out = self.act2(self.l2(out))

        return out




# myModel = model()
# x = torch.randn(32, 1, 27, 3*57)
# print(myModel(x).shape)

# ## # TESTS

# myModel = model()
# optim = torch.optim.Adam(myModel.parameters())
# # from dataLoad import getData
# # trainLoader, testLoader = getData()
# # tLoss = 0
# # for i, item in enumerate(trainLoader):
# #     loss = nn.CrossEntropyLoss()(myModel(*item[0]), item[1])
# #     loss.backward()
# #     optim.step()
# #     optim.zero_grad()
# #     tLoss += loss.item()
# #     print("{:.2f}".format(tLoss/(i+1)), end='\r')


# x1 = torch.randn(32, 1, 3, 3)
# x2 = torch.randn(32, 1, 3, 3)
# x3 = torch.randn(32, 1, 3, 3)
# print(myModel(x1, x2, x3).shape)