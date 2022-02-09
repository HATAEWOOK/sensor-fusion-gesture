"""
Network for hand rotation using mano hand model
Input : RGB(? x ?) temp 224*224
Output : rvec(3)
"""
from cmath import nan
import numpy as np
import torch
import torch.nn as nn
import sys
# sys.path.append("C:\\Users\\UVRLab\\Desktop\\sfGesture")
sys.path.append('.')
sys.path.append('..')

from utils.linear_model import LinearModel

from utils.utils_mpi_model import MANO #temp
import utils.utils_mobilenet_v3 as utils_mobilenet_v3 #tmep
from utils.resnet import resnet152 
from utils.train_utils import orthographic_proj_withz as proj



class Regressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, num_param, num_iters, max_batch_size):
        super(Regressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

        self.num_param = num_param
        self.num_iters = num_iters
        mean = np.zeros(self.num_param, dtype=np.float32)
        mean_param = np.tile(mean, max_batch_size).reshape((max_batch_size, -1)) #[bs, num_param]
        self.register_buffer('mean_param', torch.from_numpy(mean_param).float())

    def forward(self, inputs):
        """
        input : output of encoder which has ? features
        return: list of params, 
        """
        params = []
        bs = inputs.shape[0] #batch size
        param = self.mean_param[:bs, :] #[bs, num_param]

        for _ in range(self.num_iters):
            total = torch.cat([inputs, param], dim=1)
            param = param + self.fc_blocks(total)
            # param = self.fc_blocks(total)
            params.append(param)

        return params

class HMR(nn.Module):
    def __init__(self):
        super(HMR, self).__init__()

        # Number of parameters to be regressed
        # Scaling:1, Translation:2, Global rotation :3, Beta:10, Joint angle:23
        self.num_param = 39 # 1 + 2 + 3 + 10 + 23
        self.max_batch_size = 80
        self.stb_dataset = False

        # Load encoder 
        # self.encoder = utils_mobilenet_v3.mobilenetv3_small() # MobileNetV3
        # num_features = 576
        self.encoder = resnet152() # ResNet-152
        num_features = 2048
        
        # Load iterative regressor
        self.regressor = Regressor(
            fc_layers  =[num_features+self.num_param, 
                         int(num_features/2), 
                         int(num_features/2),
                         int(num_features/2),
                         self.num_param],
            use_dropout=[True,True,True,False], 
            drop_prob  =[0.5, 0.5, 0.5, 0], 
            use_ac_func=[True,True,True,False],
            num_param  =self.num_param,
            num_iters  =3,
            max_batch_size=self.max_batch_size)

        # Load MANO hand model layer
        self.mano = MANO()


    def compute_results(self, param):
        # From the input parameters [bs, num_param] 
        # Compute the resulting 2D marker location
        scale = param[:, 0].contiguous()    # [bs]    Scaling 
        trans = param[:, 1:3].contiguous()  # [bs,2]  Translation in x and y only
        rvec  = param[:, 3:6].contiguous()  # [bs,3]  Global rotation vector
        beta  = param[:, 6:16].contiguous() # [bs,10] Shape parameters
        ang   = param[:, 16:].contiguous()  # [bs,23] Angle parameters

        pose = self.mano.convert_ang_to_pose(ang)
        # rvec = torch.tanh(rvec)*np.pi
        vert, joint = self.mano(beta, pose, rvec)
        faces = self.mano.F

        # Convert from m to mm
        vert *= 1000.0
        joint *= 1000.0

        vert *= scale.unsqueeze(1).unsqueeze(2)
        joint *=scale.unsqueeze(1).unsqueeze(2)

        # For STB dataset joint 0 is at palm center instead of wrist
        # Use half the distance between wrist and middle finger MCP as palm center (root joint)
        if self.stb_dataset:
            joint[:,0,:] = (joint[:,0,:] + joint[:,9,:])/2.0
        
        # Project 3D joints to 2D image using weak perspective projection 
        # only consider x and y axis so does not rely on camera intrinsic
        # [bs,21,2] * [bs,1,1] + [bs,1,2]
        keypt = joint[:,:,:2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)

        # if self.stb_dataset: # For publication to generate images for STB dataset
        # if not self.stb_dataset:
        vert  = vert  - joint[:,9,:].unsqueeze(1) # Make all vert relative to middle finger MCP
        joint = joint - joint[:,9,:].unsqueeze(1) # Make all joint relative to middle finger MCP
            
        return keypt, joint, vert, ang, faces # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23]


    def forward(self, inputs, evaluation=True, get_feature=False):
        features = self.encoder(inputs)
        params   = self.regressor(features)

        if evaluation:
            # Only return final param
            keypt, joint, vert, ang, faces = self.compute_results(params[-1])

            if get_feature:
                return keypt, joint, vert, ang, faces, params[-1], features
            else:
                return keypt, joint, vert, ang, faces, params[-1]
        else:
            # results  = []
            # for param in params:
            #     results.append(self.compute_results(param))
            keypt, joint, vert, ang, faces = self.compute_results(params[-1])

            return keypt, joint, vert, ang, faces, params # Return the list of params at each iteration


###############################################################################
### Simple example to test the program                                      ###
###############################################################################
if __name__ == '__main__':
    def display_num_param(net):
        nb_param = 0
        for param in net.parameters():
            nb_param += param.numel()
        print('There are %d (%.2f million) parameters in this neural network' %
            (nb_param, nb_param/1e6))
    # device = torch.device('cuda' if torch.cuda.is_available() and True else 'cpu')
    model = HMR() 
    # model.load_state_dict(torch.load('C:/Users/UVRLab/Desktop/sfGesture/model/hmr_model_freihand_auc.pth'))
    # model.to(device)
    model.eval()

    # rhd = data_load_rhd_clr()
    # img, label, hand, id = rhd[7]
    # if hand is None:
    #     print(hand)
    # elif hand == 'Left':
    #     print(hand)
    # else:
    #     keypt, joint, vert, ang, params = model(img.to(device).unsqueeze(0))
    #     print(params[0][3:6])


    # display_num_param(model) # 3.82 million for MobileNetV3 Small
    # print(model.state_dict().keys())
    # model.eval()

    bs = 10 # Batch size
    image = torch.randn(bs,1,224,224)
    print(image.shape)
    keypt, joint, vert, ang, faces, params = model(image)
    print(torch.max(joint))

    print('keypt', keypt.shape) # [bs,21,2]
    print('joint', joint.shape) # [bs,21,3]
    print('vert', vert.shape) # [bs,778,3]
    print('ang', ang.shape, ang[0][0]) # [bs,23] 
    print('faces', faces.shape) #(1538,3)#
    print('params', len(params),params[0][3:6]) # 3 [bs,39] #vrec
