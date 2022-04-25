import sys
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
import torch.nn.functional
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from datetime import datetime
from celluloid import Camera
import torch.cuda.comm

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

def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    return ((im.float() / 255.0) - 0.5)


def denormalize_image(im):
    """
    float -> byte, +0.5, * pixel_max
    :param im: torch float tensor, B x C x H x W, -0.5 ~ 0.5
    :return:   torch byte tensor, B x C x H x W, 0 ~ 255
    """
    ret = (im + 0.5) * 255.0
    return ret.byte()
                

class Mano2depth():
        def __init__(self, verts, faces, joints):
                #MSRA camera intrinsic parameters
                self.w = 224
                self.h = 224
                self.verts = verts
                self.faces = faces
                self.joints = joints
                self.bs = verts.shape[0]
        
        def set_vis(self, Ks, vis):
                fx = Ks[0,0]
                fy = Ks[1,1]
                cx = Ks[0,2]
                cy = Ks[1,2]
                w = 225
                h = 225
                camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                camera_param.intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)
                vis.get_view_control().convert_from_pinhole_camera_parameters(camera_param)

                return vis
                
        def mesh2depth(self, Ks, vis, path = None):
                depths = []
                save_screen_image = True
                for vert, K in zip(self.verts, Ks):
                        vis = self.set_vis(K, vis)
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
                        else:   
                                print("depth is not captured", np.min(depth), np.max(depth))
                        # hd = HandDetector(depth, self.fx, self.fy)
                        # depth_resize, com = hd.croppedNormDepth()
                        depth_resize = cv2.resize(depth, (224,224))
                        depths.append(np.copy(depth_resize))

                if len(depths) != self.bs: return print("Error in mesh2depth")

                return torch.FloatTensor(np.asarray(depths))

        def mesh_save(self, Ks, vis, path = None):
                save_screen_image = True
                vert = self.verts[0].clone().detach().cpu()
                # faces = self.faces[0].clone().detach().cpu()
                faces = self.faces.clone().detach().cpu()
                K = Ks[0].clone().detach().cpu()
                vis = self.set_vis(K, vis)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vert)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                vis.add_geometry(mesh)
                if path is not None and save_screen_image:
                        vis.capture_screen_image(path, do_render = True)
                        save_screen_image = False
                vis.clear_geometries()
                vis.reset_view_point(True)

        def joint_save(self, Ks, vis, path = None, radius = 0.2):
                joint = self.joints[0].clone().detach().cpu()
                K = Ks[0].clone().detach().cpu()
                i=0
                vis = self.set_vis(K, vis)
                for j in joint:
                        p = o3d.geometry.TriangleMesh.create_sphere(radius)
                        p.compute_vertex_normals()
                        if i == 0:
                                p.paint_uniform_color([1,0,0])
                        elif i > 0 and i <= 4:
                                p.paint_uniform_color([1,1,0])
                        elif i > 4 and i <= 8:
                                p.paint_uniform_color([0,1,0])
                        elif i > 8 and i <= 12:
                                p.paint_uniform_color([0,1,1])
                        elif i > 12 and i <= 16:
                                p.paint_uniform_color([0,0,1])
                        elif i > 16 and i <= 20:
                                p.paint_uniform_color([0,0,0])
                        i += 1
                        p.translate(j)
                        vis.add_geometry(p)
                vis.capture_screen_image(path, do_render = True)
                vis.clear_geometries()
                vis.reset_view_point(True)

def save_image(img, path):
        img = img.squeeze().cpu()
        # img *= 100
        fig = plt.figure(1, figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        fig.savefig(path)
        plt.close(fig)

def proj_func(xyz, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    uv = torch.bmm(K,xyz.permute(0,2,1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:,:,:2]).to(device=uv.device)
    out_uv = torch.addcdiv(out_uv, uv[:,:,:2], uv[:,:,2].unsqueeze(-1).repeat(1,1,2), value=1)
    return out_uv

def orthographic_proj_withz(X, trans, scale, offset_z=0.):
    """
    X: B x N x 3
    trans: B x 2: [tx, ty]
    scale: B x 1: [sc]
    Orth preserving the z.
    """
    scale = scale.contiguous().view(-1, 1, 1)
    trans = trans.contiguous().view(scale.size(0), 1, -1)
    trans = trans[:,:,:2]

    proj = scale * X

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)

def regularizer_loss(pose):
    #tilt-swing-azimuth pose prior loss
    '''
    tsaposes: (B,15,3)
    '''
    pi = np.pi
    '''
    max_nonloss = torch.tensor([[3.15,0.01,0.01],
                                [5*pi/180,10*pi/180,100*pi/180],#0
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#6
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,pi/8,pi/8],#12
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]]).float().to(tsaposes.device)
    min_nonloss = torch.tensor([[3.13,-0.01,-0.01],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-10*pi/180]]).float().to(tsaposes.device)
    '''
    max_nonloss = torch.tensor([[5*pi/180,10*pi/180,100*pi/180],#0 INDEX
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3 MIDDLE
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,20*pi/180,100*pi/180],#6 PINKY
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9 RING
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,3*pi/16,pi/8],#12 THUMB
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]]).float().to(pose.device)
    min_nonloss = torch.tensor([[-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-20*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-20*pi/180]]).float().to(pose.device)
    pose_errors = torch.where(pose>max_nonloss.unsqueeze(0),pose-max_nonloss.unsqueeze(0),torch.zeros_like(pose)) + torch.where(pose<min_nonloss.unsqueeze(0),-pose+min_nonloss.unsqueeze(0),torch.zeros_like(pose))
    pose_loss = torch.mean(pose_errors.mul(torch.tensor([1,1,2]).float().to(pose_errors.device)))#.cpu()
    return pose_loss

def shape_loss(beta):
        shape_loss = torch.nn.functional.mse_loss(beta, torch.zeros_like(beta).to(beta.device))

        return shape_loss

def direction(j3d):
        index_vec = j3d[:, 5, :] - j3d[:, 0, :]
        mid_vec = j3d[:, 9, :] - j3d[:, 0, :]
        ring_vec = j3d[:, 13, :] - j3d[:, 0, :]
        little_vec = j3d[:, 17, :] - j3d[:, 0, :]
        hand_vec = index_vec + mid_vec + ring_vec + little_vec
        hand_direc = hand_vec / torch.norm(hand_vec, dim = 1).unsqueeze(1)

        return hand_direc

def direction_loss(target_j3d, pred_j3d):
        bs = target_j3d.shape[0]
        device = target_j3d.device
        target_direc = direction(target_j3d)
        pred_direc = direction(pred_j3d)
        # loss = torch.nn.functional.mse_loss(target_direc, pred_direc)
        inner_product = torch.bmm(target_direc.view(bs, 1, -1), pred_direc.view(bs, -1, 1)).squeeze() #cos theta
        loss = torch.nn.functional.mse_loss(inner_product, torch.ones(bs).to(device))
        
        return torch.mean(loss**2)
        
def hm_integral_loss(target_j2d, hm_2d_keypt_list):
        device = hm_2d_keypt_list[-1].device
        integral_loss = torch.zeros(1).to(device)
        for hm_2d_keypt in hm_2d_keypt_list:
                hm_2d_keypt_distance = torch.nn.functional.mse_loss(hm_2d_keypt, target_j2d)
                hm_2d_keypt_con = torch.ones_like(hm_2d_keypt_distance)
                integral_loss += torch.sum(hm_2d_keypt_distance.mul(hm_2d_keypt_con**2))/torch.sum((hm_2d_keypt_con**2))

        return integral_loss

def normalize(param, mode = 'joint'):
        if mode == 'joint':
                norm_scale = torch.norm(param[:,10,:] - param[:,9,:], dim=1) #[bs]
                param = param - param[:,9,:].unsqueeze(1)
                param = param / norm_scale.unsqueeze(1).unsqueeze(2)
        elif mode == 'keypt':
                norm_scale = torch.norm(param[:,10,:] - param[:,9,:], dim=1)
                param = param / norm_scale.unsqueeze(1).unsqueeze(2)

        return param

class ResultGif:
        def __init__(self):
                fig_3d = plt.figure(figsize = (5,5))
                fig_2d = plt.figure(figsize = (5,5))
                fig_1d = plt.figure(figsize = (5,5))
                self.camera_3d = Camera(fig_3d)
                self.camera_2d = Camera(fig_2d)
                self.camera_1d = Camera(fig_1d)
                self.ax_3d = fig_3d.add_subplot(projection='3d')
                self.ax_2d = fig_2d.add_subplot()
                self.ax_1d = fig_1d.add_subplot()

        def update(self, pred, target):
                if pred.shape[-1] == 3:
                        self.ax_3d.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker='o')
                        self.ax_3d.scatter(target[:, 0], target[:, 1], target[:, 2], marker='^')
                        self.camera_3d.snap()
                        self.ax_3d.clear()
                if pred.shape[-1] == 2:
                        self.ax_2d.scatter(pred[:, 0], pred[:, 1], c='r')
                        self.ax_2d.scatter(target[:, 0], target[:, 1], c='b')
                        self.camera_2d.snap()
                        self.ax_2d.clear()
                else:
                        self.ax_1d.scatter(target, pred, c='r')
                        self.camera_1d.snap()
                        self.ax_1d.clear()

        def save(self):
                animation_3d = self.camera_3d.animate(interval=50, blit=True)
                animation_2d = self.camera_2d.animate(interval=50, blit=True)
                animation_1d = self.camera_1d.animate(interval=50, blit=True)
                save_path = '/root/sensor-fusion-gesture/ckp/FreiHAND/logs'
                animation_3d.save(
                        # os.path.join(save_path, 'joint.gif')
                        'joint.gif'
                )
                animation_2d.save(
                        # os.path.join(save_path, 'keypt.gif')
                        'keypt.gif'
                )
                animation_1d.save(
                        # os.path.join(save_path, 'param.gif')
                        'param.gif'
                )

def compute_uv_from_integral(hm, resize_dim):
    """
    https://github.com/JimmySuen/integral-human-pose
    
    :param hm: B x K x H x W (Variable)
    :param resize_dim:
    :return: uv in resize_dim (Variable)
    
    heatmaps: C x H x W
    return: C x 3
    """
    upsample = nn.Upsample(size=resize_dim, mode='bilinear', align_corners=True)  # (B x K) x H x W
    resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1]) #[bs*21, 256, 256]
    #import pdb; pdb.set_trace()
    num_joints = resized_hm.shape[0] #bs*21
    hm_width = resized_hm.shape[-1] #256
    hm_height = resized_hm.shape[-2] #256
    hm_depth = 1
    pred_jts = softmax_integral_tensor(resized_hm, num_joints, hm_width, hm_height, hm_depth) #[1,2016]
    pred_jts = pred_jts.view(-1,hm.size(1), 3)
    #import pdb; pdb.set_trace()
    return pred_jts #[bs, 21, 3]

def softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth):
    # global soft max
    #preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = preds.reshape((1, num_joints, -1)) #[1, bs*21, 65536]
    preds = torch.nn.functional.softmax(preds, 2)
    # integrate heatmap into joint location
    x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    #x = x / float(hm_width) - 0.5
    #y = y / float(hm_height) - 0.5
    #z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))#[1,B*21,1,height,width]
    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)#[1,B*21,width=256]
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)#[1,B*21,hight=256]
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)#[1,B*21,depth=1]
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]
    accu_x = accu_x.sum(dim=2, keepdim=True) #[1,672,1]
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    #import pdb; pdb.set_trace()
    return accu_x, accu_y, accu_z



if __name__ == "__main__":
        # data = '3d'
        # data = '2d'
        data = 'pose'
        # data = 'mano'
        results = 25

        if data == '3d':
                fig = plt.figure(figsize = (5,5))
                ax_3d = fig.add_subplot(111, projection='3d')
                camera = Camera(fig)
                path = '/root/sensor-fusion-gesture/ckp/FreiHAND/results%d'%results
                for i in range(100):
                        for j in [10,20]:
                                pred_joint = np.loadtxt(os.path.join(path, 'pred', 'E%d_%d_joint.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)
                                target_joint = np.loadtxt(os.path.join(path, 'target', 'E%d_%d_joint.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)

                                ax_3d.scatter(pred_joint[:, 0], pred_joint[:, 1], pred_joint[:, 2], marker='o')
                                ax_3d.scatter(target_joint[:, 0], target_joint[:, 1], target_joint[:, 2], marker='^')
                                camera.snap()
                        
                        print(i)

                animation = camera.animate(interval=200, blit=True)
                animation.save(
                                # os.path.join(save_path, 'joint.gif')
                                'joint.gif'
                        )
        elif data == '2d':
                fig = plt.figure(figsize = (5,5))
                ax_2d = fig.add_subplot(111)
                camera = Camera(fig)
                path = '/root/sensor-fusion-gesture/ckp/FreiHAND/results%d'%results
                for i in range(100):
                        for j in [10]:
                                pred_keypt = np.loadtxt(os.path.join(path, 'pred', 'E%d_%d_keypt.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)
                                target_keypt = np.loadtxt(os.path.join(path, 'target', 'E%d_%d_keypt.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)

                                ax_2d.scatter(pred_keypt[:, 0], pred_keypt[:, 1], c='r')
                                ax_2d.scatter(target_keypt[:, 0], target_keypt[:, 1], c='b')
                                camera.snap()
                        
                        print(i)

                animation = camera.animate(interval=200, blit=True)
                animation.save(
                                # os.path.join(save_path, 'joint.gif')
                                'keypt.gif'
                        )
        elif data == 'scale':
                fig = plt.figure(figsize = (5,5))
                ax_2d = fig.add_subplot(111)
                camera = Camera(fig)
                path = '/root/sensor-fusion-gesture/ckp/FreiHAND/results%d'%results
                for i in range(100):
                        for j in [10]:
                                pred_mano = np.loadtxt(os.path.join(path, 'pred', 'E%d_%d_mano.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)
                                target_mano = np.loadtxt(os.path.join(path, 'target', 'E%d_%d_mano.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)

                                target_mano = target_mano[55:]
                                pred_mano = pred_mano[55:]
                                ax_2d.plot(np.arange(0,45), target_mano, c='r')
                                ax_2d.plot(np.arange(0,45), pred_mano, c='b')
                                ax_2d.set_yscale("log")
                                ax_2d.text(-4, 4, "%d_%d"%(i+1, j))
                                camera.snap()
                        
                        print(i)

                animation = camera.animate(interval=200, blit=True)
                animation.save(
                                # os.path.join(save_path, 'joint.gif')
                                'scale.gif'
                        )
        elif data == 'pose':
                fig = plt.figure(figsize = (5,5))
                ax_2d = fig.add_subplot(111)
                camera = Camera(fig)
                path = '/root/sensor-fusion-gesture/ckp/FreiHAND/results%d'%results
                pi = np.pi
                max_nonloss = np.array([[5*pi/180,10*pi/180,100*pi/180],#0 INDEX
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3 MIDDLE
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,20*pi/180,100*pi/180],#6 PINKY
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9 RING
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,3*pi/16,pi/8],#12 THUMB
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]])
                min_nonloss = np.array([[-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-20*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-20*pi/180]])
                for i in range(100):
                        for j in [10,20]:
                                ax_2d.plot(np.arange(0,45), max_nonloss.reshape(-1), c='r')
                                ax_2d.plot(np.arange(0,45), min_nonloss.reshape(-1), c='b')
                                pred_mano = np.loadtxt(os.path.join(path, 'pred', 'E%d_%d_mano.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)
                                target_mano = np.loadtxt(os.path.join(path, 'target', 'E%d_%d_mano.txt'%(i+1,j)), dtype=float, delimiter=' ', skiprows=0)
                                target_mano = target_mano[10:55]
                                pred_mano = pred_mano[10:55]
                                ax_2d.scatter(np.arange(0,45), target_mano, c='r')
                                ax_2d.scatter(np.arange(0,45), pred_mano, c='b')
                                ax_2d.text(-4, 4, "%d_%d"%(i+1, j))
                                camera.snap()
                        
                        print(i)

                animation = camera.animate(interval=200, blit=True)
                animation.save(
                                # os.path.join(save_path, 'joint.gif')
                                'pose.gif'
                        )
        


                