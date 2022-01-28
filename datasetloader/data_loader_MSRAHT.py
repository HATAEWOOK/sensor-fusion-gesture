from shutil import ExecError
import sys
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
import traceback
import random

import torch
from torch.utils.data import Dataset
import torchvision

import matplotlib.pyplot as plt
from PIL import Image
import struct

from utils.train_utils import Data_preprocess

def get_dataset(dat_name, base_path, queries = None, use_cache=True, train=False, split=None):
    if dat_name == 'MSRA_HT':
        hand_dataset = MSRA_HT(
            base_path = base_path,
        )
        sides = 'right'

    dataset = Dataload(
        dat_name,
        hand_dataset,
        queries=queries,
        sides = sides,
        is_train = train,
    )

    return dataset



class Dataload(Dataset):
    def __init__(self, dat_name, hand_dataset, queries=None, sides = None, is_train = 'right', ):
        self.dat_name = dat_name
        self.hand_dataset = hand_dataset
        self.queries = queries
        self.sides = sides
        self.is_train = is_train
        self.totensor = torchvision.transforms.ToTensor()
        JOINT_NUM = 21
        fx = fy = 241.42
        cx, cy = 160, 120
        cube = [200,200,200]
        self.dp = Data_preprocess(JOINT_NUM, fx, fy, cx, cy, cube)

    def __len__(self):
        return len(self.hand_dataset)

    def get_sample(self, idx, query=None):
        if query is None:
            query = self.queries
        sample = {}
        if self.dat_name == 'MSRA_HT':
            depth = self.hand_dataset.get_depth(idx)
            depth_train, _, com  = self.dp.preprocess_depth(depth)
            sample['processed'] = self.totensor(depth_train).float()
            sample['com'] = torch.FloatTensor(com)
        return sample
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx, self.queries)
        except ExecError:
            traceback.print_exc()
            print('Error at {}'.format(idx))
            random_idx = random.randint(0, len(self))
            sample = self.get_sample(random_idx, self.queries)
        return sample

class MSRA_HT:
    def __init__(self, base_path = None, set_name = None,):
        self.SUBJECT = 6
        self.base_path = base_path #D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB/Subject%d/%.6d_depth.bin
        # self.set_name = set_name
        self.load_dataset()
        self.name = 'MSRA_HT'
        
    def load_dataset(self):
        filenames = []
        for i in range(1, self.SUBJECT+1):
            dir = os.listdir(os.path.join(self.base_path, 'Subject%d' % i))
            dir = [file for file in dir if file.endswith(".bin")]
            dir = [os.path.join(self.base_path, 'Subject%d' % i, file) for file in dir]
            filenames.extend(dir)
        self.filenames = filenames
    
    def get_depth(self, idx):
        depth = np.fromfile(self.filenames[idx], dtype=np.float32)
        depth = depth.reshape(240,320)
        return depth

    def get_j3d(self, idx):
        pass

    def __len__(self):
        return len(self.filenames)



if __name__ == '__main__':
    # path = 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB'
    path = '/root/Dataset/cvpr'
    query = ['processed', 'com', 'cropped']

    tmp = get_dataset(
        'MSRA_HT',
        path,
    )
    train_size = int(0.8*len(tmp))
    test_size = len(tmp) - train_size
    train_da, test_da = torch.utils.data.random_split(tmp, [train_size, test_size])
    sample = next(iter(train_da))
    print(sample['processed'].shape)


    train_loader = torch.utils.data.DataLoader(train_da, batch_size=10,shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_da, batch_size=10,shuffle = True)

    for idx, (sample) in enumerate(train_loader):
        print(idx)
        print(sample['processed'].shape)
        break


    # JOINT_NUM = 21
    # fx = fy = 241.42
    # cx, cy = 160, 120
    # cube = [200,200,200]
    # dp = Data_preprocess(JOINT_NUM, fx, fy, cx, cy, cube)

    # tmp = MSRA_HT(base_path = path)

    # dpt = tmp.get_depth(816)
    # dpt_train, dpt_crop, com = dp.preprocess_depth(dpt)
    # print(type(dpt_train), type(dpt_crop), type(com))

    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)

    # ax1.imshow(dpt)
    # ax2.imshow(dpt_train)
    # ax3.imshow(dpt_crop)
    # print(com)
    # plt.show()

    # plt.imshow(dpt)
    # plt.show()

    # filenames = []
    # i = 1
    # dir = os.listdir(os.path.join(path, 'Subject%d' % i))
    # print(len(dir))
    # dir = [file for file in dir if file.endswith(".bin")]
    # print(dir[5])
    # dir = [os.path.join(path, 'Subject%d' % i, file) for file in dir]
    # print(dir[5])
    # depth = np.fromfile(dir[5], dtype=np.float32)
    # depth = depth.reshape(240,320)
    # print(depth.shape)