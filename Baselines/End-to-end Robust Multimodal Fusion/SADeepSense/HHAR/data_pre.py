# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
# import random



import os
import random
from os.path import join
from time import time
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.nn import BCELoss
from torch.optim import Adam
import glob
from torch.utils.data import TensorDataset, DataLoader, Dataset
#from torchvision.utils import save_image



class a:
    window_len = 512 # 512
    stride_len = 20 # 100
    # act_list = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    act_list = [0, 1, 2, 3, 4]
# act_list = [1, 2]
    # act_labels_txt = ['lay', 'sit', 'std', 'wlk', 'run', 'cyc', 'nord', 'ups', 'dws', 'vac', 'iron', 'rop']


# def prepare_data(args):
#     X=[]
#     user_labels=[]
#     act_labels=[]

#     # columns for IMU data
#     imu_locs = [4,5,6, 10,11,12, 13,14,15, 
#                 21,22,23, 27,28,29, 30,31,32, 
#                 38,39,40, 44,45,46, 47,48,49
#             ] 

#     scaler = MinMaxScaler()
#     # scaler = StandardScaler()

#     for uid in np.arange(1,10):
#         path = 'd:\\PAMAP2_Dataset\\Protocol\\subject10' + str(uid) + '.dat'
#         df = pd.read_table(path, sep=' ', header=None)
#         act_imu_filter = df.iloc[:, imu_locs] 

#         for act_id in range(len(args.act_list)):
#             act_filter =  act_imu_filter[df.iloc[:, 1] == args.act_list[act_id]]
#             act_data = act_filter.to_numpy()
                
#             act_data = np.transpose(act_data)
#             # sliding window segmentation
#             start_idx = 0
#             while start_idx + args.window_len < act_data.shape[1]:
#                 window_data = act_data[:, start_idx:start_idx+args.window_len]
#                 downsamp_data = window_data[:, ::3] # downsample from 100hz to 33.3hz
#                 downsamp_data = np.nan_to_num(downsamp_data) # remove nan

#                 X.append(downsamp_data)
#                 user_labels.append(uid)
#                 act_labels.append(act_id)
#                 start_idx = start_idx + args.stride_len

#     X_n = np.array(X).astype('float32')

#     normalized_X = np.zeros_like(X_n) # allocate numpy array for normalized data
#     for ch_id in range(X_n.shape[1]): # loop the 27 sensor channels
#         ch_data = X_n[:, ch_id, :] # the data of channel id
#         scaler = MinMaxScaler() # maybe different scalers?
#         ch_data = scaler.fit_transform(ch_data) # scale the data in this channel to [0,1]
#         normalized_X[:, ch_id, :] = ch_data # assign normalized data to normalized_X
#     #normalized_X = np.transpose(normalized_X, (0, 2, 1)) # I overwrote X here, changed dimensions into: num_samples, sequence_length, feature_length

#     normalized_X= normalized_X.reshape(normalized_X.shape[0], 1, normalized_X.shape[1], normalized_X.shape[2]) # convert list to numpy array
#     act_labels = np.array(act_labels).astype('float32')
#     act_labels = act_labels.reshape(act_labels.shape[0],1)
#     #act_labels = to_categorical(act_labels, num_classes=len(args.act_list))

#     return torch.tensor(normalized_X), torch.tensor(act_labels).squeeze().long()

# def prepare_data(args):
#     global g
#     X=[]
#     user_labels=[]
#     act_labels=[]

#     # columns for IMU data
#     imu_locs = [i for i in range(37, 102)] 

#     scaler = MinMaxScaler()
#     # scaler = StandardScaler()

#     for uid in np.arange(1,6):
#         path = 'D:\\OpportunityUCIDataset\\OpportunityUCIDataset\\dataset\\S1-ADL' + str(uid) + '.dat'
#         df = pd.read_table(path, sep=' ', header=None)
# #         print(df[243].unique())
#         df[243][df[243] == 4] = 3
#         df[243][df[243] == 5] = 4
#        # g = g or df
#        # df.iloc[:, -7][df.iloc[:, -7] == 4] = 3
#        # df.iloc[:, -7][df.iloc[:, -7] == 5] = 4
#         act_imu_filter = df.iloc[:, imu_locs] 
#         #print(act_imu_filter.shape)

#         for act_id in range(len(args.act_list)):
#             act_filter =  act_imu_filter[df.iloc[:, -7] == args.act_list[act_id]]
#             act_data = act_filter.to_numpy()
#             ##print(act_data.shape)
                
#             act_data = np.transpose(act_data)
#         #    print(act_data.shape)
#             # sliding window segmentation
#             start_idx = 0
#             while start_idx + args.window_len < act_data.shape[1]:
#                 window_data = act_data[:, start_idx:start_idx+args.window_len]
#                 downsamp_data = window_data[:, ::3] # downsample from 100hz to 33.3hz
#                 downsamp_data = np.nan_to_num(downsamp_data) # remove nan

#                 X.append(downsamp_data)
#                # print(downsamp_data.shape)
#                 user_labels.append(uid)
#                 act_labels.append(act_id)
#                 start_idx = start_idx + args.stride_len

#     X_n = np.array(X).astype('float32')
#     #print(X_n.shape)
#     normalized_X = np.zeros_like(X_n) # allocate numpy array for normalized data
#     for ch_id in range(X_n.shape[1]): # loop the 27 sensor channels
#         ch_data = X_n[:, ch_id, :] # the data of channel id
#         scaler = MinMaxScaler() # maybe different scalers?
#         ch_data = scaler.fit_transform(ch_data) # scale the data in this channel to [0,1]
#         normalized_X[:, ch_id, :] = ch_data # assign normalized data to normalized_X
#     #normalized_X = np.transpose(normalized_X, (0, 2, 1)) # I overwrote X here, changed dimensions into: num_samples, sequence_length, feature_length

#     normalized_X= normalized_X.reshape(normalized_X.shape[0], 1, normalized_X.shape[1], normalized_X.shape[2]) # convert list to numpy array
#     act_labels = np.array(act_labels).astype('float32')
#     act_labels = act_labels.reshape(act_labels.shape[0],1)
#     #act_labels = to_categorical(act_labels, num_classes=len(args.act_list))

#     return torch.tensor(normalized_X), torch.tensor(act_labels).squeeze().long()








# class myd(Dataset):
# 	def __init__(self, **kwargs):
# 		self.X, self.y = prepare_data(a)
# 		super().__init__(**kwargs)
# 		self.perm = np.random.permutation(len(self.X))
# 	def __getitem__(self, i):
# 		ind = self.perm[i]
# 		return (self.X[ind], self.y[ind])
# 	def __len__(self):
# 		return len(self.X)

SAMPLE_LENGTH = 5.0 # Length in seconds of each sample that is contained in a .csv file.
TAO = 0.25 # Interval length
NUMBER_OF_INTERVALS = int(SAMPLE_LENGTH / TAO) # It has to be in integer.
MEASUREMENTS_PER_INTERVAL = 10
FEATURE_DIM = 2 * (3 * 2) * MEASUREMENTS_PER_INTERVAL # Detailed explanation in the thesis.
OUT_DIM = 6 # We have 6 different activities.


def process(item):
    df = pd.read_csv(item, header=None)
    fileData = df.fillna(0.0).to_numpy()
    features = fileData[:, :NUMBER_OF_INTERVALS * FEATURE_DIM]
    features = np.reshape(features, [NUMBER_OF_INTERVALS, FEATURE_DIM])
    labels = fileData[:, NUMBER_OF_INTERVALS*FEATURE_DIM:]
    labels = np.reshape(labels, [OUT_DIM])
    #return (torch.tensor(features).unsqueeze(0).float(), torch.tensor(np.argmax(labels)).long())
    return (torch.tensor(features, dtype=torch.float32).unsqueeze(0), torch.tensor(np.argmax(labels), dtype=torch.long))



class myTrainDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f = glob.glob("/home/xaviar/diffusion/SADeepSense_HHAR/outputDir/*/train/*.csv")
        self.inds = np.random.permutation(len(self.f))
    
    def __getitem__(self, i):
        return process(self.f[self.inds[i]])
    
    def __len__(self):
        return len(self.f)

class myTestDataset(Dataset):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.f = glob.glob("/home/xaviar/diffusion/SADeepSense_HHAR/outputDir/*/eval/*.csv")
		self.inds = np.random.permutation(len(self.f))

	def __getitem__(self, i):
		return process(self.f[self.inds[i]])

	def __len__(self):
		return len(self.f)


def getData():
	# dataset = myDataset()
	# train_size = round(0.8*(len(dataset)))
	# test_size = len(dataset)-train_size
	# trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

	trainDataset = myTrainDataset()
    #normalize(trainDataset)
        
	testDataset = myTestDataset()
        

	trainLoader = torch.utils.data.DataLoader(trainDataset,
	    batch_size=64, shuffle=True, num_workers=8) 

	testLoader = torch.utils.data.DataLoader(testDataset,
	    batch_size=64, shuffle=True,num_workers=8) 
   
	return (trainLoader, testLoader)





