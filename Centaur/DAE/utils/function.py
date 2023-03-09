import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset

import os
from os.path import join

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from utils.sliding_window import sliding_window
import pickle as cp

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def prepare_data_OPPO(filename = '../../../../../data/oppChallenge_gestures.data'):
    # Number of Sensor Channels used in the OPPORTUNITY dataset.
    NB_SENSOR_CHANNELS = 113

    # Number of classes in which data is classified (or to be classified).
    NUM_CLASSES = 5

    # Length of the sliding window used to segmenting the time-series-data.
    SLIDING_WINDOW_LENGTH = 24

    # Steps of the sliding window used in segmenting the data.
    SLIDING_WINDOW_STEP = 12
    
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    
    assert NB_SENSOR_CHANNELS == X_train.shape[1]
    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    X_test = np.transpose(X_test, (0, 2, 1))
    X_test= X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]) # convert list to numpy array

    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    X_train = X_train.reshape((-1,SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    X_train = np.transpose(X_train, (0, 2, 1))
    X_train= X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]) # convert list to numpy array
    
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    return X_train, X_test, y_train, y_test


def prepare_dataloaders(args, X_train, X_test, y_train, y_test):
    trainDataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    testDataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    trainLoader = torch.utils.data.DataLoader(trainDataset,
        batch_size=args.batchSize, shuffle=True) 

    testLoader = torch.utils.data.DataLoader(testDataset,
        batch_size=args.batchSize, shuffle=False)
    return trainLoader, testLoader

def prepare_data_PAMAP2(args, root_path='../../../../../PAMAP2_Dataset/Protocol/subject10'):
    X=[]
    user_labels=[]
    act_labels=[]

    window_len = 512
    stride_len = 20
    # columns for IMU data
    imu_locs = [4,5,6, 10,11,12, 13,14,15, 
                21,22,23, 27,28,29, 30,31,32, 
                38,39,40, 44,45,46, 47,48,49
            ] 
    
    act_list = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    scaler = MinMaxScaler()
    # scaler = StandardScaler()

    for uid in np.arange(1,10):
        path = root_path + str(uid) + '.dat'
        df = pd.read_table(path, sep=' ', header=None)
        act_imu_filter = df.iloc[:, imu_locs] 

        for act_id in range(len(act_list)):
            act_filter =  act_imu_filter[df.iloc[:, 1] == act_list[act_id]]
            act_data = act_filter.to_numpy()
                
            act_data = np.transpose(act_data)
            # sliding window segmentation
            start_idx = 0
            while start_idx + window_len < act_data.shape[1]:
                window_data = act_data[:, start_idx:start_idx + window_len]
                downsamp_data = window_data[:, ::3] # downsample from 100hz to 33.3hz
                downsamp_data = np.nan_to_num(downsamp_data) # remove nan

                X.append(downsamp_data)
                user_labels.append(uid)
                act_labels.append(act_id)
                start_idx = start_idx + stride_len

    X_n = np.array(X).astype('float32')

    normalized_X = np.zeros_like(X_n) # allocate numpy array for normalized data
    for ch_id in range(X_n.shape[1]): # loop the 27 sensor channels
        ch_data = X_n[:, ch_id, :] # the data of channel id
        scaler = MinMaxScaler() # maybe different scalers?
        ch_data = scaler.fit_transform(ch_data) # scale the data in this channel to [0,1]
        normalized_X[:, ch_id, :] = ch_data # assign normalized data to normalized_X
        # normalized_X = np.transpose(normalized_X, (0, 2, 1)) # overwrote X here, changed dimensions into: num_samples, sequence_length, feature_length
        
    # convert list to numpy array
    normalized_X= normalized_X.reshape(normalized_X.shape[0], 1, normalized_X.shape[1], normalized_X.shape[2]) 
    act_labels = np.array(act_labels).astype('float32')
    act_labels = act_labels.reshape(act_labels.shape[0],1)
    act_labels = to_categorical(act_labels, num_classes=len(act_list))

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, act_labels, test_size=0.2, random_state=args.random_seed)
    return X_train, X_test, y_train, y_test

# def prepare_dataloaders_PAMAP2(args, X, labels):
#     dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(labels))

#     # Train/Test dataset split
#     train_size = int(args.train_split * len(dataset))
#     test_size = len(dataset) - train_size
#     trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#     trainLoader = torch.utils.data.DataLoader(trainDataset,
#         batch_size=args.batchSize, shuffle=True) 

#     testLoader = torch.utils.data.DataLoader(testDataset,
#         batch_size=args.batchSize, shuffle=False)
#     return trainLoader, testLoader

def prep_data(data, useCUDA):
	x, y = data
	if useCUDA:
		x = Variable(x.cuda())
		y = Variable(y.cuda()).view(y.size()).type_as(x)
	else:
		x = Variable(x)
		y = Variable(y).view(y.size()).type_as(x)
	return x,y

# def prep_data(data, useCUDA):
# 	x, y = data
# 	if useCUDA:
# 		x = Variable(x.cuda())
# 		y = Variable(y.cuda()).view(y.size(0),1).type_as(x)
# 	else:
# 		x = Variable(x)
# 		y = Variable(y).view(y.size(0),1).type_as(x)
# 	return x,y

def make_new_folder(exDir):
	i=1
	while os.path.isdir(join(exDir,'Ex_'+str(i))):
		i+=1

	os.mkdir(join(exDir,'Ex_'+str(i)))
	return join(exDir,'Ex_'+str(i))

def plot_losses(losses, exDir, epochs=1, title='loss'):
	#losses should be a dictionary of losses 
	# e.g. losses = {'loss1':[], 'loss2:'[], 'loss3':[], ... etc.}
	fig1 = plt.figure()
	assert epochs > 0
	for key in losses:
		noPoints = len(losses[key])
		factor = float(noPoints)/epochs
		plt.plot(np.arange(len(losses[key]))/factor,losses[key], label=key)

	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.title(title)
	fig1.savefig(join(exDir, title+'_plt.png'))

def save_input_args(exDir, opts):
	#save the input args to 
	f = open(join(exDir,'opts.txt'),'w')
	saveOpts =''.join(''.join(str(opts).split('(')[1:])\
		.split(')')[:-1])\
		.replace(',','\n')
	f.write(saveOpts)
	f.close()
    
def save_exp_details(args, model_folder):
    config_filename = 'config.txt'
    print("\nExperimental details:")

    with open(join(model_folder, config_filename), 'w') as f:
        f.writelines("Experimental details:\n")
        for attr, value in args.__dict__.items():
            print(attr, '=', value)
            f.writelines(str(attr) + '=' + str(value) + '\n')
    return


def shift_x(x, dx, dy):
	xShift = Variable(torch.Tensor(x.size()).fill_(0)).type_as(x)
	non = lambda s: s if s<0 else None
	mom = lambda s: max(0,s)

	xShift[:, :, mom(dy):non(dy), mom(dx):non(dx)] = x[:, :, mom(-dy):non(-dy), mom(-dx):non(-dx)]
	return xShift

def random_occlusion(x, size):  #TODO
	'''sqaure occlusion or WGN'''

	assert (size,size) <= x.size()

def plot_norm_losses(losses, exDir, epochs=1, title='loss'):
	#losses should be a dictionary of losses 
	# e.g. losses = {'loss1':[], 'loss2:'[], 'loss3':[], ... etc.}
	assert epochs > 0
	fig1 = plt.figure()
	for key in losses:
		y = losses[key]
		y -= np.mean(y)
		y /= ( np.std(y) + 1e-6 ) 
		noPoints = len(losses[key])
		factor = float(noPoints)/epochs
		plt.plot(np.arange(len(losses[key]))/factor, y, label=key)
	plt.xlabel('epoch')
	plt.ylabel('normalised loss')
	plt.legend()
	fig1.savefig(join(exDir, 'norm_'+title+'_plt.png'))


def binary_class_score(pred, target, thresh=0.5):
	predLabel = torch.gt(pred, thresh)
	print([predLabel, target])
	classScoreTest = torch.eq(predLabel, target.type_as(predLabel))
	return  classScoreTest.float().sum()/target.size(0)
