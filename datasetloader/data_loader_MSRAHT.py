from shutil import ExecError
import sys
sys.path.append('.')
sys.path.append('..')
import json
import os
import numpy as np
import traceback
import random

import torch
from torch.utils.data import Dataset
import torchvision

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import skimage.io as io

from utils.train_utils import Data_preprocess, Mano2depth, set_vis
from utils.hand_detector import HandDetector
from utils.utils_mpi_model import MANO
from utils import handutils

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def get_dataset(dat_name, base_path=None, queries = None, vis=None, use_cache=True, set_name=None):
    if dat_name == 'MSRA_HT':
        hand_dataset = MSRA_HT(
            base_path = base_path
        )

    if dat_name == 'synthetic':
        hand_dataset = SYN_MANO_loader(
            base_path=base_path,
        )

    if dat_name == 'FreiHAND':
        hand_dataset = FreiHAND_loader(
            base_path=base_path,
            set_name = set_name
        )


    dataset = Dataload(
        dat_name,
        hand_dataset,
        queries=queries,
        set_name = set_name
    )

    return dataset



class Dataload(Dataset):
    def __init__(self, dat_name, hand_dataset, queries=None, set_name = None):
        self.dat_name = dat_name
        self.hand_dataset = hand_dataset
        self.queries = queries
        self.set_name = set_name
        self.totensor = torchvision.transforms.ToTensor()
        # JOINT_NUM = 21
        self.fx = self.fy = 241.42
        # cx, cy = 160, 120
        # cube = [180,180,180]
        # self.dp = Data_preprocess(JOINT_NUM, fx, fy, cx, cy, cube)
        self.max_rot = np.pi

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
            sample['keypt'] = self.hand_dataset.get_j2doints(idx)
            sample['name'] = idx

        if self.dat_name == 'FreiHAND':
            image = self.hand_dataset.get_img(idx)
            if 'image' in query:
                sample['image'] = self.totensor(image).float()
            if 'maskRGBs' in query:
                maskRGB = self.hand_dataset.get_maskRGB(idx)
                sample['msakRGBs'] = self.totensor(maskRGB).float()

            K = self.hand_dataset.get_K(idx)
            if 'Ks' in query:
                sample['Ks'] = K
            if 'scales' in query:
                sample['scales'] = self.hand_dataset.get_scale(idx)
            if 'mano' in query:
                sample['mano'] = self.hand_dataset.get_mano(idx)
            if self.set_name == 'training':
                j3d = self.hand_dataset.get_j3d(idx)
                if 'j3d' in query:
                    sample['j3d'] = j3d
                verts = self.hand_dataset.get_verts(idx)
                if 'verts' in query:
                    sample['verts'] = verts
                mask = self.hand_dataset.get_mask(idx)
                if 'mask' in query:
                    sample['mask'] = torch.round(self.totensor(mask))
            open_j2d = self.hand_dataset.get_open_j2d(idx)
            if 'open_j2d' in query:
                sample['open_j2d'] = open_j2d
                sample['open_j2d_con'] = self.hand_dataset.get_open_j2d_con(idx)
            if 'cv_images' in query:
                sample['cv_images'] = self.hand_dataset.get_cv_img(idx)
            
            sample['idx'] = idx

            #Augmentation
            if self.set_name == 'training':
                if 'trans_img' in query:
                    center = np.asarray([112,112])
                    scale = 224
                    rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
                    rot_mat = np.array(
                    [
                        [np.cos(rot), -np.sin(rot), 0],
                        [np.sin(rot), np.cos(rot), 0],
                        [0, 0, 1],
                    ]
                    ).astype(np.float32)
                    affinetrans, post_rot_trans = handutils.get_affine_transform(
                        center, scale, [224, 224], rot=rot
                    )
                    trans_image = handutils.transform_img(
                        image, affinetrans, [224, 224]
                    )
                    sample['trans_img'] = self.totensor(trans_image).float()
                    sample['post_rot_trans'] = post_rot_trans
                    sample['rot'] = rot
                if 'trans_open_j2d' in query:
                    trans_open_j2d = handutils.transform_coords(open_j2d.numpy(), affinetrans)
                    sample['trans_open_j2d'] = torch.from_numpy(np.array(trans_open_j2d)).float()
                if 'trans_Ks' in query:
                    trans_Ks = post_rot_trans.dot(K)
                    sample['trans_Ks'] = torch.from_numpy(trans_Ks).float()
                if 'trans_masks' in query:
                    trans_masks = handutils.transform_img(
                        mask, affinetrans, [224,224]
                    )
                    sample['trans_mask'] = torch.round(self.totensor(trans_masks))
                if 'trans_j3d' in query:
                    trans_j3d = rot_mat.dot(
                        j3d.transpose(1,0)
                    ).transpose()
                    sample['trans_j3d'] = torch.from_numpy(trans_j3d)
                if 'trans_verts' in query:
                    trans_verts = rot_mat.dot(
                        verts.transpose(1,0)
                    ).transpose()
                    sample['trans_verts'] = torch.from_numpy(trans_verts)

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

    def get_j2doints(self, idx):
        j3d = self.get_joints(idx)
        param = self.get_params(idx)
        j2d = j3d[:,:,:2] * param[0].unsqueeze(0).unsqueeze(1) + param[1:3].unsqueeze(0).unsqueeze(1)
        return j2d


    def __len__(self):
        return int(len(os.listdir(self.base_path)) / 3)

class FreiHAND_loader:
    def __init__(self, base_path = None, set_name = None,):
        self.base_path = base_path #/root/Dataset/FreiHAND_pub_v2
        self.openpose_path = '/root/Dataset/openpose'
        self.set_name = set_name
        self.load_dataset()
        self.name = 'FreiHAND'
        
    def load_dataset(self):
        if self.set_name == 'evaluation':
            set_name = 'evaluation'
        else:
            set_name = 'training'

        self.K_list = json_load(os.path.join(self.base_path, '%s_K.json'%set_name))
        self.scale_list = json_load(os.path.join(self.base_path, '%s_scale.json'%set_name))
        prefix_template = "{:08d}"
        idxs = sorted([int(imgname.split(".")[0]) for imgname in os.listdir(os.path.join(self.base_path, set_name, 'rgb'))])
        self.prefixs = [prefix_template.format(idx) for idx in idxs]
        del idxs
        self.open_j2d_lists = json_load(os.path.join(self.openpose_path, set_name, 'detect_%s.json'%set_name))
        self.open_j2d_list = self.open_j2d_lists[0]
        self.open_j2d_con_list = self.open_j2d_lists[1]

        if set_name == 'training':
            self.mano_list = json_load(os.path.join(self.base_path, 'training_mano.json'))
            self.verts_list = json_load(os.path.join(self.base_path, 'training_verts.json'))
            self.j3d_list = json_load(os.path.join(self.base_path, 'training_xyz.json'))
    
    def get_img(self, idx):
        img_path = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        img = Image.open(img_path).convert('RGB')
        return img

    def get_cv_img(self, idx):
        img_path = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        cv_img = cv2.imread(img_path)
        return cv_img

    def get_mask(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        mask_path = os.path.join(self.base_path, self.set_name, 'mask', '{}.jpg'.format(self.prefixs[idx]))
        mask = Image.open(mask_path)
        return mask

    def get_maskRGB(self, idx):
        img_path = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        img = io.imread(img_path)
        if idx >= 32560:
            idx = idx % 32560
            mask_path = os.path.join(self.base_path, self.set_name, 'mask', '{}.jpg'.format(self.prefixs[idx]))
        mask_img = io.imread(mask_path, 1)
        mask_img = np.rint(mask_img)
        img[~mask_img.astype(bool)] = 0
        return img
        
    def get_K(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        K = torch.FloatTensor(np.array(self.K_list[idx]))
        return K

    def get_scale(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        scale = self.scale_list[idx]
        return scale

    def get_mano(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        mano = torch.FloatTensor(self.mano_list[idx])
        return mano

    def get_j3d(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        joint = torch.FloatTensor(self.j3d_list[idx])
        return joint

    def get_verts(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        verts = torch.FloatTensor(self.verts_list[idx])
        return verts

    def get_open_j2d(self, idx):
        open_j2d = torch.FloatTensor(self.open_j2d_list[idx])
        return open_j2d

    def get_open_j2d_con(self, idx):
        open_j2d_con = torch.FloatTensor(self.open_j2d_con_list[idx])
        return open_j2d_con

    def __len__(self):
        return len(self.prefixs)

if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # import torch.utils
    import math
    queries = ['mano']
    dat = get_dataset('FreiHAND' ,base_path='/root/Dataset/FreiHAND_pub_v2', queries=queries, set_name = 'training')
    sample = next(iter(dat))
    print(sample['mano'].shape)






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

    # vis = set_vis()
    # dat = SYN_MANO_generator(data_size=10000, save_path = '/root/Dataset/synthetic/003', vis = vis)
    # dat.generate_file()
