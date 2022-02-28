from ast import expr_context
from operator import indexOf
from pkgutil import get_data
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

from utils.train_utils import Data_preprocess, Mano2depth, set_vis
from utils.hand_detector import HandDetector
from utils.utils_mpi_model import MANO

def get_dataset(dat_name, base_path=None, queries = None, vis=None, use_cache=True, train=False, split=None):
    if dat_name == 'MSRA_HT':
        hand_dataset = MSRA_HT(
            base_path = base_path
        )

    if dat_name == 'synthetic':
        hand_dataset = SYN_MANO_loader(
            base_path=base_path,
        )


    dataset = Dataload(
        dat_name,
        hand_dataset,
        queries=queries,
    )

    return dataset



class Dataload(Dataset):
    def __init__(self, dat_name, hand_dataset, queries=None,):
        self.dat_name = dat_name
        self.hand_dataset = hand_dataset
        self.queries = queries
        self.totensor = torchvision.transforms.ToTensor()
        # JOINT_NUM = 21
        self.fx = self.fy = 241.42
        # cx, cy = 160, 120
        # cube = [180,180,180]
        # self.dp = Data_preprocess(JOINT_NUM, fx, fy, cx, cy, cube)

    def __len__(self):
        return len(self.hand_dataset)

    def get_sample(self, idx, query=None):
        if query is None:
            query = self.queries
        sample = {}
        if self.dat_name == 'MSRA_HT':
            depth = self.hand_dataset.get_depth(idx)
            hd = HandDetector(depth, self.fx, self.fy)
            depth_processed, com = hd.croppedNormDepth()
            sample['processed'] = self.totensor(depth_processed).float()
            sample['com'] = torch.FloatTensor(com)
            sample['name'] = self.hand_dataset.get_filename(idx)
            sample['j3d'] = self.totensor(self.hand_dataset.get_j3d(idx)).float()
            sample['j2d'] = self.totensor(self.hand_dataset.get_j2d(idx)).float()
        
        if self.dat_name == 'synthetic':
            sample['depth'] = self.hand_dataset.get_depth(idx)
            sample['params'] = self.hand_dataset.get_params(idx)
            sample['joints'] = self.hand_dataset.get_joints(idx)
            sample['name'] = idx

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
        jointlists = np.empty([0,21,3])
        for i in range(1, self.SUBJECT+1):
            dir = os.listdir(os.path.join(self.base_path, 'Subject%d' % i))
            dir = [file for file in dir if file.endswith(".bin")]
            dir = [os.path.join(self.base_path, 'Subject%d' % i, file) for file in dir]

            joint_path = os.path.join(self.base_path, 'Subject%d' % i, 'joint.txt')
            jointlist = np.loadtxt(joint_path, dtype=float, delimiter=' ', skiprows=1)
            jointlist = [joint.reshape([-1,3]) for joint in jointlist] #[400,21,3]
            jointlists=np.append(jointlists, jointlist, axis=0)
            filenames.extend(dir)
        self.filenames = filenames
        self.jointlists = jointlists
    
    def get_depth(self, idx):
        depth = np.fromfile(self.filenames[idx], dtype=np.float32)
        depth = depth.reshape(240,320)
        return depth

    def get_filename(self, idx):
        i = idx // 400
        if i == 0:
            name = f'S{i+1}_id{idx}'
        else:
            name = f'S{i+1}_id{idx % (400 * i)}'
        return name

    def get_j3d(self, idx):
        j3d = self.jointlists[idx]
        j3d = np.insert(j3d, 1, j3d[17:21], axis=0)
        j3d = np.delete(j3d, slice(21,25), axis=0)
        j3d = j3d - j3d[9,:]
        return j3d

    def get_j2d(self, idx):
        j3d = self.get_j3d(idx)
        j2d = j3d[:,:2]
        return j2d

    def __len__(self):
        return len(self.filenames)

class SYN_MANO_generator:
    def __init__(self, data_size = 5000, save_path = None, vis = None):
        self.data_size = data_size
        self.save_path = save_path 
        mano = MANO()
        plim = mano.plim_
        alim = mano.alim_
        self.random_parameters(alim)
        self.name = 'SYN_MANO'
        self.vis = vis
        self.mano = mano

        
    def random_parameters(self, alim):
        params = []
        for _ in range(self.data_size):
            scale = torch.randn(1)
            trans = torch.randn(2)
            rot = torch.randn(3)
            beta = torch.randn(10)
            # theta = torch.randn(45) #ang 23
            # plim = plim.view(45,-1)
            # theta = torch.zeros([45])
            # for i in range(len(theta)):
            #     theta[i] = np.random.uniform(plim[i,0], plim[i,1])

            # params.append(torch.cat((scale, trans, rot, beta, theta)))

            ang = torch.zeros([23])
            for i in range(len(ang)):
                ang[i] = np.random.uniform(alim[i,0], alim[i,1])

            params.append(torch.cat((scale, trans, rot, beta, ang)))

        self.params = params
    
    def generate_gt(self, idx):
        verts, faces, joints = self.to_mano(self.params[idx])
        m2d = Mano2depth(verts, faces)
        depth = m2d.mesh2depth(self.vis)
        return depth, joints

    def get_params(self, idx):
        # np.savetxt(os.path.join(self.save_path, '%d_params.txt'%idx), self.params[idx].numpy())
        return self.params[idx]

    def generate_file(self):
        for idx in range(self.data_size):
            depth, joints = self.generate_gt(idx)
            torch.save(depth, os.path.join(self.save_path, '%d_depth.pt'%idx))
            torch.save(joints, os.path.join(self.save_path, '%d_joints.pt'%idx))
            params = self.get_params(idx)
            torch.save(params, os.path.join(self.save_path, '%d_params.pt'%idx))
            if (idx+1) % 10 == 0:
                print("%d / %d"%(idx+1, self.data_size))

    # def to_mano(self, params):
    #     scale = params[0].unsqueeze(0)
    #     trans = params[1:3].unsqueeze(0)
    #     rot = params[3:6].unsqueeze(0)
    #     beta = params[6:16].unsqueeze(0)
    #     theta = params[16:61].view(-1,3).unsqueeze(0)

    #     verts, joints = self.mano(beta, theta, rot)
    #     faces = torch.tensor(self.mano.F)
    #     verts *= 1000.0
    #     joints *= 1000.0

    #     joints = joints - joints[:,9,:].unsqueeze(1)

    #     # verts[:,:,:2] = verts[:,:,:2]*scale.unsqueeze(0).unsqueeze(1) + trans.unsqueeze(0)

    #     return verts, faces, joints

    def to_mano(self, params):
        scale = params[0].unsqueeze(0)
        trans = params[1:3].unsqueeze(0)
        rot = params[3:6].unsqueeze(0)
        beta = params[6:16].unsqueeze(0)
        ang = params[16:].unsqueeze(0)

        pose = self.mano.convert_ang_to_pose(ang)
        verts, joints = self.mano(beta, pose, rot)
        faces = torch.tensor(self.mano.F)
        verts *= 1000.0
        joints *= 1000.0

        verts  = verts  - joints[:,9,:].unsqueeze(1)
        joints = joints - joints[:,9,:].unsqueeze(1)

        # verts[:,:,:2] = verts[:,:,:2]*scale.unsqueeze(0).unsqueeze(1) + trans.unsqueeze(0)

        return verts, faces, joints

    def __len__(self):
        return self.data_size

class SYN_MANO_loader:
    def __init__(self, base_path = None):
        self.base_path = base_path # /root/Dataset/synthetic/001
        self.name = 'SYN_MANO'
    
    def get_depth(self, idx):
        depth = torch.load(os.path.join(self.base_path, '%d_depth.pt'%idx))
        return depth

    def get_params(self, idx):
        params = torch.load(os.path.join(self.base_path, '%d_params.pt'%idx))
        return params

    def get_joints(self, idx):
        joints = torch.load(os.path.join(self.base_path, '%d_joints.pt'%idx))
        return joints

    def __len__(self):
        return int(len(os.listdir(self.base_path)) / 3)

if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # import torch.utils
    # dat = get_dataset('synthetic' ,base_path='/root/Dataset/synthetic/001')
    # kwargs = {
    #         'num_workers' : 4,
    #         'batch_size' : 64,
    #         'shuffle' : True,
    #         'drop_last' : True,
    #     }

    # train_size = int(0.8*len(dat))
    # test_size = len(dat) - train_size
    # train_data, test_data = torch.utils.data.random_split(dat, [train_size, test_size])
    # train_dataloader = DataLoader(train_data, **kwargs)
    # test_dataloader = DataLoader(test_data, **kwargs)
    # print("Data size : ",len(train_dataloader), len(test_dataloader))

    # sample = next(iter(train_dataloader))
    # print(sample['depth'].shape)

    vis = set_vis()
    dat = SYN_MANO_generator(data_size=10000, save_path = '/root/Dataset/synthetic/003', vis = vis)
    dat.generate_file()
