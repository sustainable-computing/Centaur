
import torch
from torch import nn
from os.path import join
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler, TensorDataset
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, BatchNorm1d, Dropout, Flatten, BCELoss

# from torchsummary import summary


class HAR(nn.Module):
    
    def __init__(self, n_sensor_channels=113, len_seq=24, n_hidden=128, n_layers=1, n_filters=64, 
                 n_classes=5, filter_size=(1,5), drop_prob=0.5):
        super(HAR, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.n_sensor_channels = n_sensor_channels
        self.len_seq = len_seq

        self.conv1 = nn.Conv2d(1, n_filters, filter_size)
        self.conv2 = nn.Conv2d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv2d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv2d(n_filters, n_filters, filter_size)
        
        # self.lstm1  = nn.LSTM(64, n_hidden, n_layers)
        # self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=n_sensor_channels*n_filters, num_heads=1) # 7232=113*64
        # self.fc0 = nn.Linear(57856, 128)
        self.fc = nn.Linear(n_sensor_channels*n_filters*(len_seq-4*(filter_size[1]-1)), n_classes) #57856 = 8*113*64

        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        # x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH,1) # for direct channel_gate
        # batch_size = x.shape[0]

        # x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH) # for deepconvlstm conv layers
        # x = torch.permute(x,(1,0,2))
        # print(x.shape)
        # x, attn_output_weights = self.multihead_attn0(x,x,x)

        # print(x.shape)
        # x = torch.permute(x,(2,1,0))
        # print(x.shape)
        # x = x.view(-1, 1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH) # draft
        # x = torch.permute(x, (0,2,1))
        # x = torch.unsqueeze(x, dim=1)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # [64, 113, 8]
        # x = x.view(-1, NB_SENSOR_CHANNELS, 8, 1)
        # x = x.view(x.shape[0], x.shape[1], x.shape[2], 1)
        # x = x.view(x.shape[0], -1, 8)
        
        # print(x.shape)
        x = torch.permute(x, (3,0,1,2))
        x = x.view(x.shape[0], x.shape[1],-1)
        
        # print(x.shape)
        # x = x.view(8, x.shape[0], -1) # bak
        
    
        x, attn_output_weights = self.multihead_attn(x,x,x)
        x = self.dropout(x)
        x = F.relu(x)
#         x, attn_output_weights = self.multihead_attn1(x,x,x)
#         # x = self.dropout(x)
#         x = F.relu(x)    
        
        x = torch.permute(x, (1,0,2))
        
        # x, hidden = self.lstm1(x, hidden)
        # # x = self.dropout(x)
        # x, hidden = self.lstm2(x, hidden)
        # x = self.dropout(x)
        
        # x = x.contiguous().view(-1, self.n_hidden)

        x = torch.reshape(x, (x.shape[0],-1))
        # x = F.relu(self.fc0(x))
        # x = self.dropout(x)
        x = self.fc(x)
        
        # out = x.view(x.shape[0], -1, self.n_classes)[:,-1,:]
        return x
    
#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
#         train_on_gpu = torch.cuda.is_available()

#         if (train_on_gpu):
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
#         return hidden

def get_eval_model(n_sensor_channels, len_seq, num_classes, model_path):
    model = HAR(n_sensor_channels=n_sensor_channels, len_seq=len_seq, n_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

