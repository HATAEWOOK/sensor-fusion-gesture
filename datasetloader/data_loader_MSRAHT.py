from ast import expr_context
from operator import indexOf
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
        cube = [180,180,180]
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
            sample['name'] = self.hand_dataset.get_filename(idx)
            sample['j3d'] = self.totensor(self.hand_dataset.get_j3d(idx)).float()
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
        name = self.filenames[idx]
        return name

    def get_j3d(self, idx):
        j3d = self.jointlists[idx]
        j3d = np.insert(j3d, 1, j3d[17:21], axis=0)
        j3d = np.delete(j3d, slice(21,25), axis=0)
        return j3d

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    import open3d as o3d
    from utils.utils_mpi_model import MANO


    path = 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB'
    # # path = '/root/Dataset/cvpr'
    # query = ['processed', 'com', 'cropped']

    # tmp = get_dataset(
    #     'MSRA_HT',
    #     path,
    # )
    # print(len(tmp))
    # train_size = int(0.8*len(tmp))
    # test_size = len(tmp) - train_size
    # train_da, test_da = torch.utils.data.random_split(tmp, [train_size, test_size])
    # sample = next(iter(train_da))
    # joints = sample['j3d'].squeeze()
    # print(joints.shape)
    # print(sample['name'])

    # bs = 1 # Batchsize
    # beta = torch.zeros([bs,10], dtype=torch.float32)
    # rvec = torch.zeros([bs,3], dtype=torch.float32)
    # tvec = torch.zeros([bs,3], dtype=torch.float32)
    # pose = torch.zeros([bs,15,3], dtype=torch.float32)
    # ppca = torch.zeros([bs,45], dtype=torch.float32)
    # beta = torch.randn(bs,10)
    # rvec = torch.randn(bs,3)
    # tvec = torch.randn(bs,3)
    # pose = torch.randn(bs,15,3)
    # ppca = torch.randn(bs,45)
    # mano = MANO()
    # verts, joints_mano = mano(beta, pose, rvec, tvec)
    # faces = mano.F
    verts = 'D:/sfGesture/ckp/vert/E  1_ 10_vert.txt'
    faces = 'D:/sfGesture/ckp/results/E  1_ 10_faces.txt'
    verts = np.loadtxt(verts, dtype=float, delimiter=' ', skiprows=0)
    faces = np.loadtxt(faces, dtype=float, delimiter=' ', skiprows=0)
    # joints_mano *= 2

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    joint_path = os.path.join(path, 'Subject5', 'joint.txt')
    jointlist = np.loadtxt(joint_path, dtype=float, delimiter=' ', skiprows=1)
    jointlist = [joint.reshape([-1,3]) for joint in jointlist]
    joints = jointlist[154]

    joints = np.insert(joints, 1, joints[17:21], axis=0)
    joints = np.delete(joints, slice(21,25), axis=0)
    # joints[:,2] -= 100
    # print(np.mean(joints[:,2]-joints_mano[:,2]))
    joints[:,2] += 350
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
    mesh_spheres = []
    # joints[:,2] *= 0
    # joints_mano[0,:,2] *= 0 
    i=0
    for j in joints:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=2)
        m.compute_vertex_normals()
        if i == 0:
            m.paint_uniform_color([1,0,0])
        elif i > 0 and i <= 4:
            m.paint_uniform_color([1,1,0])
        elif i > 4 and i <= 8:
            m.paint_uniform_color([0,1,0])
        elif i > 8 and i <= 12:
            m.paint_uniform_color([0,1,1])
        elif i > 12 and i <= 16:
            m.paint_uniform_color([0,0,1])
        elif i > 16 and i <= 20:
            m.paint_uniform_color([0,0,0])
        i += 1
        m.translate(j)
        mesh_spheres.append(m)
    # joints[:,2] *= 0.1
    # i=0
    # for j in joints_mano:
    #     m1 = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    #     m1.compute_vertex_normals()
    #     if i == 0:
    #         m1.paint_uniform_color([1,0,0])
    #     elif i > 0 and i <= 4:
    #         m1.paint_uniform_color([1,1,0])
    #     elif i > 4 and i <= 8:
    #         m1.paint_uniform_color([0,1,0])
    #     elif i > 8 and i <= 12:
    #         m1.paint_uniform_color([0,1,1])
    #     elif i > 12 and i <= 16:
    #         m1.paint_uniform_color([0,0,1])
    #     elif i > 16 and i <= 20:
    #         m1.paint_uniform_color([0,0,0])
    #     i += 1
    #     m1.translate(j)
    #     mesh_spheres.append(m1)

    # i=0
    # for j in depth_joint:
    #     m2 = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    #     m2.compute_vertex_normals()
    #     if i == 0:
    #         m2.paint_uniform_color([1,0,0])
    #     elif i > 0 and i <= 4:
    #         m2.paint_uniform_color([1,1,0])
    #     elif i > 4 and i <= 8:
    #         m2.paint_uniform_color([0,1,0])
    #     elif i > 8 and i <= 12:
    #         m2.paint_uniform_color([0,1,1])
    #     elif i > 12 and i <= 16:
    #         m2.paint_uniform_color([0,0,1])
    #     elif i > 16 and i <= 20:
    #         m2.paint_uniform_color([0,0,0])
    #     i += 1
    #     m2.translate(j)
    #     mesh_spheres.append(m2)

    tmp = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    tmp.compute_vertex_normals()
    tmp.paint_uniform_color([0,0,0])
    tmp.translate([0,0,0])
    # o3d.visualization.draw_geometries([mesh])

    w = 320
    h = 240
    fx = fy = 241.42
    cx = w / 2 - 0.5
    cy = h / 2 - 0.5
    cube = [180,180,180]
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible = False, width = w, height = h)
    vc = vis.get_view_control()
    camera_param = vc.convert_to_pinhole_camera_parameters()
    camera_param.intrinsic.set_intrinsics(w, h, fx, fy, w / 2 - 0.5,  h / 2 - 0.5)
    vc.convert_from_pinhole_camera_parameters(camera_param)
    vis.add_geometry(mesh)
    path = 'D:/sfGesture/01.png'
    path2 = 'D:/sfGesture/02.png'
    vis.capture_screen_image(path, do_render = True)
    vis.capture_depth_image(path2, do_render = True)
    # vis.run()





    # train_loader = torch.utils.data.DataLoader(train_da, batch_size=10,shuffle = True)
    # test_loader = torch.utils.data.DataLoader(test_da, batch_size=10,shuffle = True)



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