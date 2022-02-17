import sys
from cv2 import Param_MAT
sys.path.append('.')
sys.path.append('..')
# sys.path.append("C:\\Users\\UVRLab\\Desktop\\sfGesture")
import numpy as np
import cv2
from scipy import stats, ndimage
import itertools
from torch.utils.data import Subset
import logging
import open3d as o3d
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime

from utils.utils_mpi_model import MANO
from utils.hand_detector import HandDetector

class Data_preprocess():
        def __init__(self, jointNum, fx, fy, cx, cy, cube):
                self.joint = jointNum
                self.fx = fx
                self.fy = fy
                self.cx = cx
                self.cy = cy
                self.calibMat = np.asarray([[self.fx, 0, self.cx], [0, self.fy, self.cy],[0,0,1]])
                self.cube = cube
                self.rng = np.random.RandomState(23455)

        def preprocess_depth(self, depth_orig):
                depth_orig[depth_orig>500]=0
                print("ckp1")
                com = self.calculateCOM(depth_orig)
                print("ckp2")
                com, depth_crop, window = self.refineCOMIterative(depth_orig, com, 3)
                depth_train = self.makeLearningImage(depth_orig, com)
                        
                return depth_train, depth_crop, com

        def makeLearningImage(self,img_crop,com):
                s=224
                cnnimg=img_crop.copy()
                cnnimg[cnnimg==0]=com[2]+self.cube[2]/2.
                cnnimg=cnnimg-com[2]
                cnnimg=cnnimg/(self.cube[2]/2.)

                cnnimg=cv2.resize(cnnimg,(s,s))
                return np.copy(cnnimg)

        def refineCOMIterative(self,dimg,com,num_iter):
                dpt=dimg.copy()
                for k in range(num_iter):
                        #size=np.asarray(size)*(1-0.1*k)
                        #print(size)
                        xstart, xend, ystart, yend, zstart, zend=self.comToBounds(com,self.cube)
                        
                        xstart=max(xstart,0)
                        ystart=max(ystart,0)
                        xend=min(xend,dpt.shape[1])
                        yend=min(yend,dpt.shape[0])

                        cropped=self.crop(dpt,xstart,xend,ystart,yend,zstart,zend)
                        
                        com =self.calculateCOM(cropped)
                        
                        if np.allclose(com,0.):
                                com[2]=cropped[cropped.shape[0]//2,cropped.shape[1]//2]
                        com[0]+=max(xstart,0)
                        com[1]+=max(ystart,0)

                return com,cropped,[xstart,xend,ystart,yend,zstart,zend] 

        def crop(self,dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
                cropped = dpt[ystart:yend, xstart:xend].copy()
                msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
                msk2 = np.bitwise_and(cropped > zend, cropped != 0)
                cropped[msk1] = zstart
                cropped[msk2] = 0.
                return cropped

        def comToBounds(self,com,size):
                '''
                com: [pixel,pixel,mm] 
                '''
                fx=self.fx
                fy=self.fy
                
                zstart = com[2] - size[2] / 2.
                zend = com[2] + size[2] / 2
                xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
                xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
                ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
                yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))
                
                return xstart, xend, ystart, yend, zstart, zend

        def calculateCOM(self,dimg,minDepth=10,maxDepth=1000):
                
                dc=dimg.copy()

                dc[dc<minDepth]=0
                dc[dc>maxDepth]=0

                cc=ndimage.measurements.center_of_mass(dc>0) #0.001

                num=np.count_nonzero(dc) #0.0005
                
                com=np.array((cc[1]*num,cc[0]*num,dc.sum()),np.float64) #0.0002
                
                if num==0:
                        raise Exception('com can not be calculated (calculateCOM)')  
                else:
                        return com/num

class ConcatDataLoader:
        def __init__(self, dataloaders):
                self.loaders = dataloaders
        
        def __iter__(self):
                self.iters = [iter(loader) for loader in self.loaders]
                self.idx_cycle = itertools.cycle(list(range(len(self.loaders))))
                return self
        
        def __next__(self):
                loader_idx = next(self.idx_cycle)
                loader = self.iters[loader_idx]
                batch = next(loader)
                if isinstance(loader.dataset, Subset):
                        dataset = loader.dataset.dataset
                else:
                        dataset = loader.dataset
                dat_name = dataset.hand_dataset.dat_name
                batch["dataset"] = dat_name
                batch["root"] = "wrist"
                batch["use_streohands"] = True
                batch["split"] = dataset.pose_dataset.split

                return batch

        def __len__(self):
                return sum(len(loader) for loader in self.loaders)

def mklogger(log_path, mode = 'w'):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        log_path = os.path.join(log_path, str(datetime.now().replace(microsecond=0)) + ".log") 
        fh = logging.FileHandler('%s'%log_path, mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

class AverageMeter():
        def __init__(self):
                self.reset()

        def reset(self):
                self.val = 0
                self.avg = 0 
                self.sum = 0 
                self.count = 0

        def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count

def set_vis():
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
        camera_param.intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)
        vc.convert_from_pinhole_camera_parameters(camera_param)

        return vis

class Mano2depth():
        def __init__(self, verts, faces):
                #MSRA camera intrinsic parameters
                self.w = 320
                self.h = 240
                self.fx = self.fy = 241.42
                self.cx = self.w / 2 - 0.5
                self.cy = self.h / 2 - 0.5
                self.cube = [180,180,180]
                self.verts = verts
                self.faces = faces.detach().cpu()
                self.bs = verts.shape[0]
                
        def mesh2depth(self, vis, path = None):
                depths = []
                save_screen_image = True
                for vert in self.verts:
                        mesh = o3d.geometry.TriangleMesh()
                        vert = vert.detach().cpu()
                        mesh.vertices = o3d.utility.Vector3dVector(vert)
                        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
                        mesh.compute_vertex_normals()
                        vis.add_geometry(mesh)
                        if path is not None and save_screen_image:
                                vis.capture_screen_image(path, do_render = True)
                                save_screen_image = False
                        depth = vis.capture_depth_float_buffer(do_render =True)
                        vis.clear_geometries()
                        vis.reset_view_point(True)
                        depth = np.asarray(depth)
                        if np.max(depth) != 0:
                                depth /= np.max(depth)
                        # else:   
                        #         print("depth is not captured", np.min(depth), np.max(depth))
                        # hd = HandDetector(depth, self.fx, self.fy)
                        # depth_resize, com = hd.croppedNormDepth()
                        depth_resize = cv2.resize(depth, (224,224))
                        depths.append(np.copy(depth_resize))

                if len(depths) != self.bs: return print("Error in mesh2depth")

                return torch.FloatTensor(np.asarray(depths))

def save_image(img, path):
        img = img.squeeze().cpu()
        # img *= 100
        fig = plt.figure(1, figsize=[6, 6])
        plt.imshow(img, cmap='gray')
        plt.imshow(img)
        plt.axis('off')
        fig.savefig(path)
        plt.close(fig)

def orthographic_proj_withz(X, trans, scale, offset_z=0.):
    """
    X: B x N x 3
    trans: B x 2: [tx, ty]
    scale: B x 1: [sc]
    Orth preserving the z.
    """
    scale = scale.contiguous().view(-1, 1, 1)
    trans = trans.contiguous().view(scale.size(0), 1, -1)

    proj = scale * X

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)

def regularizer_loss(ang, theta=None):
        ang = ang.detach().cpu()
        mano = MANO()
        limits = []

        for i in range(ang.shape[0]):
                if theta is None:
                        limit = mano.compute_ang_limit(ang[i])
                else:
                        limit = mano.compute_pose_limit(ang[i])
                limits.append(limit.detach())
        
        loss = np.asarray(limits)

        return np.mean(loss)

def normalize(values, dim = '3d'):
    values = values.detach()

    for i in range(values.shape[0]):
        if dim == '3d':
            meanx, meany, meanz = torch.mean(values[i,:,0]),torch.mean(values[i,:,1]),torch.mean(values[i,:,2])
            stdx, stdy, stdz = torch.std(values[i,:,0]),torch.std(values[i,:,1]),torch.std(values[i,:,2])

            if stdx == 0: 
                values[i,:,0] = 0 
            else: 
                values[i,:,0]=(values[i,:,0] - meanx) / stdx
            if stdy == 0: 
                values[i,:,1] = 0 
            else: 
                values[i,:,1]=(values[i,:,1] - meany) / stdy
            if stdz == 0: 
                values[i,:,2] = 0 
            else: 
                values[i,:,2]=(values[i,:,2] - meanz) / stdz


        if dim == '2d':
            meanx, meany = torch.mean(values[i,:,0]),torch.mean(values[i,:,1])
            stdx, stdy = torch.std(values[i,:,0]),torch.std(values[i,:,1])

            if stdx == 0: 
                values[i,:,0] = 0 
            else: 
                values[i,:,0]=(values[i,:,0] - meanx) / stdx
            if stdy == 0: 
                values[i,:,1] = 0 
            else: 
                values[i,:,1]=(values[i,:,1] - meany) / stdy

    return values


if __name__ == "__main__":
        bs = 10 # Batchsize
        beta = torch.zeros([bs,10], dtype=torch.float32)
        rvec = torch.zeros([bs,3], dtype=torch.float32)
        # rvec = torch.tensor([[np.pi/2,0,0]], dtype=torch.float32)
        tvec = torch.zeros([bs,3], dtype=torch.float32)

        model = MANO()
        pose = torch.zeros([bs,15,3], dtype=torch.float32)
        ppca = torch.zeros([bs,45], dtype=torch.float32)
        # pose = model.convert_pca_to_pose(ppca)
        vertices, joints = model.forward(beta, pose, rvec, tvec)
        m2d = Mano2depth(vertices, model.F)
        pred = m2d.mesh2depth()
        print(pred.shape) #[bs, 224, 224]


                