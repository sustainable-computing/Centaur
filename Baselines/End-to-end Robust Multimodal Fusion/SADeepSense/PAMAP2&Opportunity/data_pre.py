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

def prepare_data(args):
    global g
    X=[]
    user_labels=[]
    act_labels=[]

    # columns for IMU data?
    imu_locs = [i for i in range(37, 102)] 

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    for uid in np.arange(1,6):
        path = '.../OpportunityUCIDataset/dataset/S1-ADL' + str(uid) + '.dat'
        df = pd.read_table(path, sep=' ', header=None)
#         print(df[243].unique())
        df[243][df[243] == 4] = 3
        df[243][df[243] == 5] = 4
       # g = g or df
       # df.iloc[:, -7][df.iloc[:, -7] == 4] = 3
       # df.iloc[:, -7][df.iloc[:, -7] == 5] = 4
        act_imu_filter = df.iloc[:, imu_locs] 
        #print(act_imu_filter.shape)

        for act_id in range(len(args.act_list)):
            # -7?
            act_filter =  act_imu_filter[df.iloc[:, -7] == args.act_list[act_id]]
            act_data = act_filter.to_numpy()
            ##print(act_data.shape)
                
            act_data = np.transpose(act_data)
        #    print(act_data.shape)
            # sliding window segmentation
            start_idx = 0
            while start_idx + args.window_len < act_data.shape[1]:
                window_data = act_data[:, start_idx:start_idx+args.window_len]
                downsamp_data = window_data[:, ::3] # downsample from 100hz to 33.3hz
                downsamp_data = np.nan_to_num(downsamp_data) # remove nan

                X.append(downsamp_data)
               # print(downsamp_data.shape)
                user_labels.append(uid)
                act_labels.append(act_id)
                start_idx = start_idx + args.stride_len

    X_n = np.array(X).astype('float32')
    #print(X_n.shape)
    normalized_X = np.zeros_like(X_n) # allocate numpy array for normalized data
    for ch_id in range(X_n.shape[1]): # loop the 27 sensor channels
        ch_data = X_n[:, ch_id, :] # the data of channel id
        scaler = MinMaxScaler() # maybe different scalers?
        ch_data = scaler.fit_transform(ch_data) # scale the data in this channel to [0,1]
        normalized_X[:, ch_id, :] = ch_data # assign normalized data to normalized_X
    #normalized_X = np.transpose(normalized_X, (0, 2, 1)) # I overwrote X here, changed dimensions into: num_samples, sequence_length, feature_length

    normalized_X= normalized_X.reshape(normalized_X.shape[0], 1, normalized_X.shape[1], normalized_X.shape[2]) # convert list to numpy array
    act_labels = np.array(act_labels).astype('float32')
    act_labels = act_labels.reshape(act_labels.shape[0],1)
    #act_labels = to_categorical(act_labels, num_classes=len(args.act_list))

    return torch.tensor(normalized_X), torch.tensor(act_labels).squeeze().long()








class myd(Dataset):
	def __init__(self, **kwargs):
		self.X, self.y = prepare_data(a)
		super().__init__(**kwargs)
		self.perm = np.random.permutation(len(self.X))
	def __getitem__(self, i):
		ind = self.perm[i]
		return (self.X[ind], self.y[ind])
	def __len__(self):
		return len(self.X)


def getData():
	dataset = myd()
	train_size = round(0.8*(len(dataset)))
	test_size = len(dataset)-train_size
	trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

	trainLoader = torch.utils.data.DataLoader(trainDataset,
	    batch_size=32, shuffle=True) 


	testLoader = torch.utils.data.DataLoader(testDataset,
	    batch_size=32, shuffle=True) 
	return (trainLoader, testLoader)





# MEAN_OF_IMU = [-0.32627436907665514, -0.8661114601303396]
# STD_OF_IMU = [0.6761486428324216, 113.55369543559192]
# MEAN_OF_SKELETON = [-0.08385579666058844, -0.2913725901521685, 2.8711066708996738]
# STD_OF_SKELETON = [0.14206656362043646, 0.4722835954035046, 0.16206781976658088]

# random.seed(0)


# class Multimodal_dataset():
# 	"""Build dataset from motion sensor data."""
# 	def __init__(self, x1, x2, y):

# 		self.data1 = x1.tolist() #concate and tolist
# 		self.data2 = x2.tolist() #concate and tolist
# 		self.labels = y.tolist() #tolist

# 		self.data1 = torch.tensor(self.data1) # to tensor
# 		self.data2 = torch.tensor(self.data2) # to tensor
# 		self.labels = torch.tensor(self.labels)
# 		self.labels = (self.labels).long()


# 	def __len__(self):
# 		return len(self.labels)

# 	def __getitem__(self, idx):

# 		sensor_data1 = self.data1[idx]
# 		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

# 		sensor_data2 = self.data2[idx]
# 		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

# 		activity_label = self.labels[idx]

# 		return sensor_data1, sensor_data2, activity_label


# class Unimodal_dataset():
# 	"""Build dataset from motion sensor data."""
# 	def __init__(self, x, y):

# 		self.data = x.tolist() #concate and tolist
# 		self.labels = y.tolist() #tolist

# 		self.data = torch.tensor(self.data) # to tensor
# 		self.labels = torch.tensor(self.labels)
# 		self.labels = (self.labels).long()


# 	def __len__(self):
# 		return len(self.labels)

# 	def __getitem__(self, idx):

# 		sensor_data = self.data[idx]
# 		sensor_data = torch.unsqueeze(sensor_data, 0)

# 		activity_label = self.labels[idx]

# 		return sensor_data, activity_label


		
# def load_class_data_single(sensor_str, activity_class, train_test_flag, label_rate):

# 	data_all_subject = []

# 	NUM_OF_TRAIN_SUBJECT = 6

# 	## lode labeled data in different labelling rate according to the "label_file_name"
# 	if label_rate == 5:
# 		train_folder = 'label-27-5-percent'
# 		label_file_name = ['a1_s6_t3', 'a2_s5_t4', 'a3_s3_t4', 'a4_s2_t2', 'a5_s4_t4', 'a6_s3_t2', 'a7_s5_t4', 'a8_s2_t3', 'a9_s3_t1', 
# 		'a10_s4_t3', 'a11_s6_t2', 'a12_s4_t4', 'a13_s2_t3', 'a14_s5_t1', 'a15_s4_t2', 'a16_s2_t4', 'a17_s6_t1', 'a18_s6_t2', 
# 		'a19_s5_t4', 'a20_s6_t1', 'a21_s2_t3', 'a22_s5_t1', 'a23_s6_t3', 'a24_s5_t4', 'a25_s3_t2', 'a26_s1_t1', 'a27_s3_t2']
# 	elif label_rate == 10:
# 		train_folder = 'label-54-10-percent'
# 		label_file_name = ['a1_s6_t3', 'a1_s5_t4', 'a2_s3_t1', 'a2_s2_t3', 'a3_s4_t3', 'a3_s3_t3', 'a4_s5_t4', 'a4_s2_t3', 'a5_s3_t4', 
# 		'a5_s4_t3', 'a6_s6_t3', 'a6_s4_t2', 'a7_s2_t3', 'a7_s5_t2', 'a8_s4_t3', 'a8_s2_t2', 'a9_s6_t1', 'a9_s5_t1', 'a10_s5_t3', 'a10_s6_t3', 
# 		'a11_s2_t2', 'a11_s5_t1', 'a12_s6_t4', 'a12_s5_t4', 'a13_s3_t4', 'a13_s1_t4', 'a14_s3_t4', 'a14_s4_t4', 'a15_s6_t3', 'a15_s6_t2', 'a16_s3_t3',
# 		 'a16_s6_t2', 'a17_s2_t4', 'a17_s5_t4', 'a18_s4_t4', 'a18_s1_t3', 'a19_s5_t4', 'a19_s3_t3', 'a20_s5_t2', 'a20_s5_t3', 'a21_s1_t4', 'a21_s3_t4',
# 		 'a22_s6_t4', 'a22_s2_t1', 'a23_s2_t3', 'a23_s6_t2', 'a24_s2_t3', 'a24_s4_t4', 'a25_s2_t1', 'a25_s6_t3', 'a26_s5_t1', 'a26_s3_t1', 'a27_s1_t4', 'a27_s2_t2']
# 	elif label_rate == 15:
# 		train_folder = 'label-81-15-percent'
# 		label_file_name = ['a1_s6_t4', 'a1_s5_t3', 'a1_s3_t2', 'a2_s2_t3', 'a2_s4_t2', 'a2_s3_t4', 'a3_s5_t4', 'a3_s2_t4', 'a3_s3_t3', 'a4_s4_t4', 
# 		'a4_s6_t3', 'a4_s4_t2', 'a5_s2_t3', 'a5_s5_t4', 'a5_s4_t4', 'a6_s2_t4', 'a6_s6_t1', 'a6_s6_t3', 'a7_s5_t2', 'a7_s6_t3', 'a7_s2_t4', 'a8_s5_t1', 
# 		'a8_s6_t3', 'a8_s4_t1', 'a9_s3_t1', 'a9_s1_t4', 'a9_s3_t2', 'a10_s4_t4', 'a10_s6_t1', 'a10_s5_t1', 'a11_s3_t3', 'a11_s6_t1', 'a11_s2_t3', 
# 		'a12_s5_t4', 'a12_s4_t3', 'a12_s1_t3', 'a13_s5_t1', 'a13_s3_t3', 'a13_s5_t3', 'a14_s5_t3', 'a14_s1_t2', 'a14_s3_t2', 'a15_s6_t4', 'a15_s2_t1', 
# 		'a15_s1_t1', 'a16_s6_t4', 'a16_s2_t1', 'a16_s4_t1', 'a17_s2_t1', 'a17_s6_t4', 'a17_s5_t4', 'a18_s3_t1', 'a18_s1_t2', 'a18_s2_t1', 'a19_s4_t2', 
# 		'a19_s6_t1', 'a19_s1_t3', 'a20_s4_t2', 'a20_s5_t1', 'a20_s4_t3', 'a21_s5_t1', 'a21_s4_t1', 'a21_s6_t4', 'a22_s4_t1', 'a22_s4_t2', 'a22_s3_t3', 
# 		'a23_s4_t4', 'a23_s3_t4', 'a23_s4_t1', 'a24_s2_t3', 'a24_s2_t4', 'a24_s2_t1', 'a25_s4_t3', 'a25_s4_t4', 'a25_s3_t2', 'a26_s1_t2', 'a26_s5_t3', 
# 		'a26_s6_t2', 'a27_s6_t1', 'a27_s6_t2', 'a27_s5_t2']
# 	elif label_rate == 20:
# 		train_folder = 'label-108-20-percent'
# 		label_file_name = ['a1_s2_t3', 'a1_s1_t1', 'a1_s3_t1', 'a1_s1_t2', 'a2_s1_t1', 'a2_s3_t4', 'a2_s6_t1', 'a2_s5_t3', 
# 		'a3_s5_t4', 'a3_s2_t2', 'a3_s4_t4', 'a3_s3_t2', 'a4_s2_t1', 'a4_s1_t2', 'a4_s3_t1', 'a4_s6_t2', 'a5_s5_t1', 'a5_s5_t4', 
# 		'a5_s6_t4', 'a5_s2_t2', 'a6_s2_t1', 'a6_s4_t2', 'a6_s5_t1', 'a6_s6_t1', 'a7_s6_t1', 'a7_s1_t4', 'a7_s4_t1', 'a7_s5_t1', 
# 		'a8_s4_t4', 'a8_s2_t3', 'a8_s3_t4', 'a8_s1_t2', 'a9_s6_t4', 'a9_s6_t3', 'a9_s4_t4', 'a9_s2_t3', 'a10_s6_t2', 'a10_s4_t1', 
# 		'a10_s6_t1', 'a10_s6_t4', 'a11_s4_t4', 'a11_s3_t4', 'a11_s4_t2', 'a11_s3_t3', 'a12_s1_t4', 'a12_s2_t3', 'a12_s5_t4', 'a12_s1_t3', 
# 		'a13_s1_t3', 'a13_s4_t3', 'a13_s2_t2', 'a13_s4_t4', 'a14_s3_t4', 'a14_s3_t1', 'a14_s6_t4', 'a14_s2_t4', 'a15_s3_t3', 'a15_s2_t1', 
# 		'a15_s4_t4', 'a15_s3_t1', 'a16_s3_t4', 'a16_s5_t3', 'a16_s2_t1', 'a16_s4_t1', 'a17_s6_t3', 'a17_s1_t3', 'a17_s2_t3', 'a17_s2_t4', 
# 		'a18_s5_t1', 'a18_s4_t1', 'a18_s2_t4', 'a18_s2_t1', 'a19_s2_t4', 'a19_s3_t1', 'a19_s1_t4', 'a19_s5_t4', 'a20_s6_t4', 'a20_s6_t3', 
# 		'a20_s5_t4', 'a20_s6_t1', 'a21_s1_t3', 'a21_s2_t2', 'a21_s6_t4', 'a21_s5_t4', 'a22_s3_t2', 'a22_s6_t3', 'a22_s4_t1', 'a22_s5_t1', 
# 		'a23_s2_t3', 'a23_s2_t2', 'a23_s3_t4', 'a23_s1_t3', 'a24_s3_t1', 'a24_s6_t2', 'a24_s2_t4', 'a24_s1_t3', 'a25_s1_t1', 'a25_s2_t4', 
# 		'a25_s5_t4', 'a25_s3_t1', 'a26_s2_t3', 'a26_s1_t2', 'a26_s6_t4', 'a26_s3_t2', 'a27_s2_t1', 'a27_s1_t2', 'a27_s1_t4', 'a27_s3_t1']
# 	elif label_rate == 30:
# 		train_folder = 'label-162-30-percent'
# 		label_file_name = ['a1_s2_t4', 'a1_s1_t4', 'a1_s3_t3', 'a1_s1_t1', 'a1_s3_t1', 'a1_s5_t1', 'a2_s6_t4', 'a2_s5_t3', 'a2_s5_t1', 
# 		'a2_s2_t1', 'a2_s4_t3', 'a2_s2_t3', 'a3_s2_t3', 'a3_s1_t4', 'a3_s2_t1', 'a3_s6_t1', 'a3_s5_t4', 'a3_s5_t1', 'a4_s5_t4', 'a4_s2_t1', 
# 		'a4_s2_t4', 'a4_s4_t4', 'a4_s6_t4', 'a4_s6_t3', 'a5_s6_t4', 'a5_s1_t1', 'a5_s4_t3', 'a5_s5_t2', 'a5_s4_t4', 'a5_s2_t4', 'a6_s3_t2', 
# 		'a6_s1_t3', 'a6_s6_t1', 'a6_s6_t2', 'a6_s4_t3', 'a6_s2_t2', 'a7_s6_t4', 'a7_s4_t3', 'a7_s6_t1', 'a7_s6_t2', 'a7_s4_t4', 'a7_s3_t3', 
# 		'a8_s4_t1', 'a8_s3_t4', 'a8_s1_t3', 'a8_s2_t1', 'a8_s5_t3', 'a8_s1_t2', 'a9_s1_t4', 'a9_s4_t2', 'a9_s2_t1', 'a9_s5_t3', 'a9_s3_t4', 
# 		'a9_s3_t1', 'a10_s6_t1', 'a10_s2_t3', 'a10_s3_t4', 'a10_s4_t2', 'a10_s2_t4', 'a10_s3_t3', 'a11_s3_t4', 'a11_s5_t1', 'a11_s2_t4', 
# 		'a11_s4_t1', 'a11_s6_t2', 'a11_s1_t4', 'a12_s1_t2', 'a12_s2_t4', 'a12_s5_t1', 'a12_s4_t4', 'a12_s2_t1', 'a12_s3_t2', 'a13_s2_t2', 
# 		'a13_s3_t2', 'a13_s1_t3', 'a13_s5_t4', 'a13_s6_t3', 'a13_s6_t1', 'a14_s5_t2', 'a14_s6_t3', 'a14_s1_t2', 'a14_s2_t3', 'a14_s6_t2', 
# 		'a14_s5_t1', 'a15_s3_t1', 'a15_s6_t4', 'a15_s4_t2', 'a15_s5_t4', 'a15_s2_t4', 'a15_s2_t3', 'a16_s3_t3', 'a16_s1_t3', 'a16_s3_t2', 
# 		'a16_s6_t4', 'a16_s2_t2', 'a16_s1_t4', 'a17_s1_t2', 'a17_s2_t4', 'a17_s5_t4', 'a17_s3_t3', 'a17_s2_t2', 'a17_s1_t1', 'a18_s6_t4', 
# 		'a18_s3_t2', 'a18_s2_t3', 'a18_s1_t1', 'a18_s2_t2', 'a18_s2_t1', 'a19_s5_t4', 'a19_s1_t1', 'a19_s1_t4', 'a19_s3_t2', 'a19_s2_t3', 
# 		'a19_s6_t2', 'a20_s1_t3', 'a20_s4_t1', 'a20_s5_t2', 'a20_s3_t4', 'a20_s6_t1', 'a20_s3_t3', 'a21_s2_t2', 'a21_s3_t2', 'a21_s1_t1', 
# 		'a21_s3_t1', 'a21_s2_t4', 'a21_s6_t4', 'a22_s5_t4', 'a22_s3_t4', 'a22_s1_t4', 'a22_s2_t1', 'a22_s2_t3', 'a22_s2_t2', 'a23_s2_t3', 
# 		'a23_s6_t3', 'a23_s1_t4', 'a23_s2_t4', 'a23_s6_t1', 'a23_s4_t3', 'a24_s6_t3', 'a24_s3_t3', 'a24_s6_t4', 'a24_s4_t4', 'a24_s5_t2', 
# 		'a24_s5_t4', 'a25_s3_t2', 'a25_s1_t3', 'a25_s2_t4', 'a25_s6_t3', 'a25_s6_t2', 'a25_s6_t1', 'a26_s3_t2', 'a26_s4_t3', 'a26_s5_t4', 
# 		'a26_s4_t4', 'a26_s5_t2', 'a26_s4_t2', 'a27_s4_t4', 'a27_s5_t2', 'a27_s2_t2', 'a27_s6_t1', 'a27_s6_t2', 'a27_s1_t3']
# 	elif label_rate == 40:
# 		train_folder = 'label-216-40-percent'
# 		label_file_name = ['a1_s2_t1', 'a1_s1_t3', 'a1_s3_t4', 'a1_s2_t4', 'a1_s1_t2', 'a1_s4_t4', 'a1_s6_t4', 'a1_s5_t1', 'a2_s5_t4', 
# 		'a2_s2_t1', 'a2_s4_t2', 'a2_s2_t4', 'a2_s2_t2', 'a2_s1_t4', 'a2_s3_t2', 'a2_s6_t4', 'a3_s5_t1', 'a3_s6_t2', 'a3_s5_t2', 'a3_s2_t2', 
# 		'a3_s2_t3', 'a3_s4_t4', 'a3_s5_t3', 'a3_s6_t1', 'a4_s6_t2', 'a4_s1_t3', 'a4_s4_t2', 'a4_s5_t3', 'a4_s5_t2', 'a4_s2_t1', 'a4_s3_t1', 
# 		'a4_s1_t4', 'a5_s6_t2', 'a5_s6_t4', 'a5_s4_t4', 'a5_s2_t3', 'a5_s6_t3', 'a5_s4_t3', 'a5_s1_t2', 'a5_s3_t3', 'a6_s4_t2', 'a6_s3_t4', 
# 		'a6_s5_t3', 'a6_s4_t4', 'a6_s1_t4', 'a6_s2_t3', 'a6_s5_t2', 'a6_s1_t1', 'a7_s1_t4', 'a7_s4_t2', 'a7_s2_t3', 'a7_s4_t1', 'a7_s3_t1', 
# 		'a7_s5_t2', 'a7_s6_t4', 'a7_s2_t1', 'a8_s3_t4', 'a8_s2_t2', 'a8_s4_t3', 'a8_s3_t3', 'a8_s4_t4', 'a8_s5_t1', 'a8_s1_t2', 'a8_s5_t4', 
# 		'a9_s6_t1', 'a9_s1_t3', 'a9_s1_t2', 'a9_s2_t2', 'a9_s5_t1', 'a9_s4_t1', 'a9_s2_t4', 'a9_s3_t4', 'a10_s2_t4', 'a10_s3_t4', 'a10_s1_t4', 
# 		'a10_s5_t1', 'a10_s6_t3', 'a10_s6_t2', 'a10_s5_t3', 'a10_s6_t4', 'a11_s1_t4', 'a11_s2_t4', 'a11_s6_t1', 'a11_s5_t3', 'a11_s3_t3', 
# 		'a11_s6_t3', 'a11_s4_t3', 'a11_s5_t4', 'a12_s2_t2', 'a12_s2_t4', 'a12_s3_t2', 'a12_s1_t3', 'a12_s3_t4', 'a12_s6_t3', 'a12_s3_t3', 
# 		'a12_s1_t1', 'a13_s1_t2', 'a13_s2_t3', 'a13_s5_t4', 'a13_s3_t4', 'a13_s2_t2', 'a13_s4_t2', 'a13_s6_t4', 'a13_s3_t2', 'a14_s2_t2', 
# 		'a14_s1_t1', 'a14_s4_t1', 'a14_s2_t3', 'a14_s5_t3', 'a14_s1_t3', 'a14_s1_t2', 'a14_s3_t4', 'a15_s2_t3', 'a15_s6_t1', 'a15_s1_t1', 
# 		'a15_s4_t4', 'a15_s5_t4', 'a15_s3_t4', 'a15_s6_t3', 'a15_s3_t2', 'a16_s2_t1', 'a16_s3_t2', 'a16_s1_t1', 'a16_s3_t4', 'a16_s2_t4', 
# 		'a16_s6_t4', 'a16_s5_t1', 'a16_s3_t1', 'a17_s1_t2', 'a17_s2_t4', 'a17_s2_t1', 'a17_s3_t4', 'a17_s2_t3', 'a17_s6_t3', 'a17_s1_t4', 
# 		'a17_s4_t1', 'a18_s6_t3', 'a18_s4_t1', 'a18_s6_t1', 'a18_s3_t1', 'a18_s6_t2', 'a18_s5_t2', 'a18_s5_t1', 'a18_s5_t4', 'a19_s3_t2', 
# 		'a19_s1_t4', 'a19_s2_t3', 'a19_s6_t3', 'a19_s6_t4', 'a19_s5_t2', 'a19_s3_t4', 'a19_s4_t3', 'a20_s5_t3', 'a20_s4_t3', 'a20_s6_t4', 
# 		'a20_s5_t4', 'a20_s1_t3', 'a20_s5_t1', 'a20_s2_t4', 'a20_s6_t2', 'a21_s6_t1', 'a21_s1_t1', 'a21_s6_t4', 'a21_s6_t2', 'a21_s5_t2', 
# 		'a21_s1_t3', 'a21_s2_t4', 'a21_s1_t2', 'a22_s6_t2', 'a22_s5_t2', 'a22_s1_t1', 'a22_s2_t4', 'a22_s4_t1', 'a22_s4_t4', 'a22_s5_t1', 
# 		'a22_s6_t1', 'a23_s2_t4', 'a23_s1_t4', 'a23_s6_t1', 'a23_s1_t3', 'a23_s6_t2', 'a23_s4_t2', 'a23_s5_t1', 'a23_s5_t4', 'a24_s1_t1', 
# 		'a24_s4_t3', 'a24_s6_t4', 'a24_s2_t4', 'a24_s5_t2', 'a24_s3_t4', 'a24_s2_t1', 'a24_s5_t3', 'a25_s3_t1', 'a25_s4_t3', 'a25_s1_t4', 
# 		'a25_s1_t2', 'a25_s5_t4', 'a25_s3_t4', 'a25_s6_t3', 'a25_s4_t1', 'a26_s1_t3', 'a26_s3_t2', 'a26_s5_t4', 'a26_s4_t3', 'a26_s1_t2', 
# 		'a26_s6_t4', 'a26_s5_t3', 'a26_s2_t3', 'a27_s4_t4', 'a27_s3_t4', 'a27_s5_t4', 'a27_s2_t4', 'a27_s2_t3', 'a27_s3_t3', 'a27_s6_t3', 'a27_s1_t3']


# 	if train_test_flag == 1:#train label

# 		file_per_class = int(label_rate / 5)

# 		for file_id in range( file_per_class ):

# 			temp_file = label_file_name[ activity_class * file_per_class + file_id ]#0-> 0,1; 1-> 2,3

# 			data_sample = np.load('../UTD-data/split-6-2/train/' + train_folder + '/label/' + sensor_str + '/' + temp_file + '_' + sensor_str + '.npy')
# 			data_all_subject.extend(data_sample)

# 	elif train_test_flag == 2:#test

# 		for subject_id in range(2):

# 			for test_id in range(4):

# 				temp_file = 'a' + str(activity_class+1) + '_s' + str(subject_id + 7) + '_t' + str(test_id + 1)

# 				# except a8_s1_t4_depth, a23_s6_t4_depth, a27_s8_t4_depth
# 				if temp_file == 'a8_s1_t4' or temp_file == 'a23_s6_t4' or temp_file == 'a27_s8_t4':
# 					print("No such file:", temp_file)
# 				else:
# 					data_sample = np.load('../UTD-data/split-6-2/test/' + sensor_str + '/' + temp_file + '_' + sensor_str + '.npy')
# 					data_all_subject.extend(data_sample)

# 	elif train_test_flag == 3:#train unlabel

# 		for subject_id in range(NUM_OF_TRAIN_SUBJECT):

# 			for test_id in range(4):

# 				temp_file = 'a' + str(activity_class+1) + '_s' + str(subject_id + 1) + '_t' + str(test_id + 1)

# 				# except a8_s1_t4_depth, a23_s6_t4_depth, a27_s8_t4_depth
# 				if temp_file == 'a8_s1_t4' or temp_file == 'a23_s6_t4' or temp_file == 'a27_s8_t4':
# 					print("No such file:", temp_file)
# 				elif (temp_file in label_file_name) == False:
# 					data_sample = np.load('../UTD-data/split-6-2/train/' + train_folder + '/unlabel/' + sensor_str + '/' + temp_file + '_' + sensor_str + '.npy')
# 					data_all_subject.extend(data_sample)

# 	data_all_subject = np.array(data_all_subject)

# 	return data_all_subject




# def sensor_data_normalize(sensor_str, data):

# 	if sensor_str == 'inertial':
# 		data[:,:,0:3] = (data[:,:,0:3] - MEAN_OF_IMU[0]) / STD_OF_IMU[0]
# 		data[:,:,3:6] = (data[:,:,3:6] - MEAN_OF_IMU[1]) / STD_OF_IMU[1]

# 	elif sensor_str == 'skeleton':
# 		for axis_id in range(3):
# 			data[:,:,:,axis_id] = (data[:,:,:,axis_id] - MEAN_OF_SKELETON[axis_id]) / STD_OF_SKELETON[axis_id]

# 	return data


# def load_data(num_of_total_class, num_per_class, train_test_flag, label_rate):

# 	x1 = []
# 	x2 = []
# 	y = []

# 	for class_id in range(num_of_total_class):

# 		data_all_subject_1 = load_class_data_single('inertial', class_id, train_test_flag, label_rate).reshape(-1, 120, 6)
# 		data_all_subject_2 = load_class_data_single('skeleton', class_id, train_test_flag, label_rate).reshape(-1, 20, 3, 40)

# 		class_all_num_data = data_all_subject_1.shape[0]
# 		label_all_subject = np.ones(class_all_num_data) * class_id

# 		# random sample data
# 		if class_all_num_data < num_per_class[class_id]:
# 			num_per_class[class_id] = class_all_num_data
# 		sample_index = random.sample(range(0, class_all_num_data), num_per_class[class_id])

# 		temp_data_1 = data_all_subject_1[sample_index]
# 		temp_data_2 = data_all_subject_2[sample_index]
# 		temp_label= label_all_subject[sample_index]

# 		x1.extend(temp_data_1)
# 		x2.extend(temp_data_2)
# 		y.extend(temp_label)

# 	x1 = np.array(x1)
# 	x2 = np.array(x2)
# 	y = np.array(y)

# 	x2 = x2.swapaxes(1,3).swapaxes(2,3)#(-1, 40, 20, 3)

# 	x1 = sensor_data_normalize('inertial', x1)
# 	x2 = sensor_data_normalize('skeleton', x2)

# 	print(x1.shape)
# 	print(x2.shape)
# 	print(y.shape)

# 	return x1, x2, y


