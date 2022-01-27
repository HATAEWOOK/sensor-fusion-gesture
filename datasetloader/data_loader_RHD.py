import sys
sys.path.append("C:\\Users\\UVRLab\\Desktop\\sfGesture")

import numpy as np
import os
import torch
from torchaudio import transforms
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
import pickle
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

from utils.hand_detector import hand_detector

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

class data_load_rhd_clr():
    '''
    320 x 320
    '''
    def __init__(self, data_path = 'D:/datasets/RHD_v1-1/RHD_published_v2', set = 'training'):
        self.data = []
        self.ids = []
        self.totensor = torchvision.transforms.ToTensor()
        self.data_path = os.path.join(data_path, set)

        with open(os.path.join(self.data_path, 'anno_%s.pickle' % set), 'rb') as fi:
            self.labels = pickle.load(fi)

        self.data_path = os.path.join(self.data_path, 'color')
        filenames = os.listdir(self.data_path)
        self.data = [os.path.join(self.data_path, filename) for filename in filenames] #image path
        self.ids = [filename for filename in filenames]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        detector = hand_detector(con=0.8)

        id = self.ids[index]
        label = self.labels[index]
        image = cv2.imread(image_path)
        hand_list = detector.detect(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(hand_list) == 0:
            return torch.zeros([3,100,100]), label, None, id #hand is not found
        
        hand1 = hand_list[0]
        x, y, w, h = hand1[0]
        w += 60
        h += 30
        cx, cy = hand1[1]
        handType = hand1[2]

        if len(hand_list) == 2:
            pass

        image=self.totensor(image)
        image = tf.crop(image, cy-h//2, cx-w//2, h, w)
        # image = tf.normalize(image,(0.3161, 0.2755, 0.2586), (0.1862, 0.1620, 0.1687), inplace=True) #necessary?
        image = tf.resize(image, (224,224))

        return image, label, handType, id


class data_load_rhd_dpt():
    '''
    320 x 320
    '''
    def __init__(self, data_path = 'D:/datasets/RHD_v1-1/RHD_published_v2', set='training'):
        self.data = []
        self.ids = []
        self.totensor = torchvision.transforms.ToTensor()
        self.data_path = os.path.join(data_path, set)

        with open(os.path.join(self.data_path, 'anno_%s.pickle' % set), 'rb') as fi:
            self.labels = pickle.load(fi)

        filenames = os.listdir(os.path.join(self.data_path, 'depth'))
        self.data = [os.path.join(self.data_path,'depth', filename) for filename in filenames] #image path
        self.ids = [filename for filename in filenames]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        detector = hand_detector(con=0.8)

        id = self.ids[index]
        label = self.labels[index]
        image = cv2.imread(image_path)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = depth_two_uint8_to_float(image[:,:,0], image[:,:,1]) * 100 # depth in cm unit

        clr_image = cv2.imread(os.path.join(self.data_path, 'color', id))
        hand_list = detector.detect(clr_image)
        
        if len(hand_list) == 0:
            return torch.zeros([3,100,100]), label, None, id #hand is not found
        
        hand1 = hand_list[0]
        x, y, w, h = hand1[0]
        w += 60
        h += 30
        cx, cy = hand1[1]
        handType = hand1[2]

        if len(hand_list) == 2:
            pass

        image=self.totensor(image)
        image = tf.crop(image, cy-h//2, cx-w//2, h, w)
        # image /= 500
        image = tf.resize(image, (224,224))

        return image, label, handType, id
        


if __name__ == '__main__':
    tfpil = torchvision.transforms.ToPILImage()
    rhd_clr = data_load_rhd_clr()
    rhd_depth = data_load_rhd_dpt()

    img, label, _, _ = rhd_depth[7]
    tmp = img[0, 66:160,120:196]
    tmp[tmp >= 500] = torch.nan
    tmp = np.array(tmp)

    print(np.nanmin(tmp))
    print(np.nanmax(tmp))

    # plt.figure()
    # plt.imshow(img[0])
    # plt.show()

    # camera_intrinsic_matrix = label['K'] # matrix containing intrinsic parameters
    # print(R.from_matrix(camera_intrinsic_matrix).as_euler('xyz', degrees=True))
    # print(img)


    # pixels = []

    # for idx in range(len(rhd_clr)):
    #     img, _, hand,_ = rhd_clr[idx]
    #     try:
    #         if hand is None:
    #             continue
    #         pixels.append(img[2].contiguous().view(-1))
    #     except TypeError:
    #         print(f'{id} is truncated')
    #     # if idx % 500 == 0:
    #     #     print(f'{idx}/41258')
    #     if idx == 500:
    #         break  # Out of memory..

    # pixels = torch.cat(pixels, dim=0)
    # std, mean = torch.std_mean(pixels, dim=0)
    # print(std, mean) #(0.1862, 0.1620, 0.1687), (0.3161, 0.2755, 0.2586)


    # coord_uv = label['uv_vis'][:, :2] # u, v coordinates of 42 hand keypoints, pixel
    # visible = label['uv_vis'][:, 2] == 1 # visibility of the keypoints, boolean
    # coord_xyz = label['xyz'] # x, y, z coordinates of the keypoints, in meters
    # camera_intrinsic_matrix = label['K'] # matrix containing intrinsic parameters
    # # print(R.from_matrix(camera_intrinsic_matrix).as_rotvec())

    # # Project world coordinates into the camera frame
    # coord_uv_proj = np.matmul(coord_xyz, np.transpose(camera_intrinsic_matrix))
    # coord_uv_proj = coord_uv_proj[:, :2] / coord_uv_proj[:, 2:]

    #     # Visualize data
    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222, projection='3d')

    # ax1.imshow(depth.permute(1,2,0))
    # ax1.plot(coord_uv[visible, 0], coord_uv[visible, 1], 'ro')
    # ax1.plot(coord_uv_proj[visible, 0], coord_uv_proj[visible, 1], 'gx')
    # ax2.scatter(coord_xyz[visible, 0], coord_xyz[visible, 1], coord_xyz[visible, 2])
    # ax2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    # ax2.set_ylabel('y')
    # ax2.set_xlabel('x')
    # ax2.set_zlabel('z')

    # plt.show()








