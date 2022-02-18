"""
Network for hand rotation using mano hand model
Input : RGB(? x ?) temp 224*224
Output : rvec(3)
"""
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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return torch.flatten(x,1)


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
    def __init__(self, cfg):
        super(HMR, self).__init__()

        # Number of parameters to be regressed
        # Scaling:1, Translation:2, Global rotation :3, Beta:10, Joint angle:23
        self.num_param = 39 # 1 + 2 + 3 + 10 + 23
        self.max_batch_size = 80
        self.stb_dataset = False

        # Load encoder 
        # self.encoder = utils_mobilenet_v3.mobilenetv3_small() # MobileNetV3
        # num_features = 576
        if cfg.pretrained:
            self.encoder = resnet152(pretrained=True)
        else:
            self.encoder = resnet152()
        
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = Identity()

        num_features = 2048
        
        # Load iterative regressor
        self.regressor = Regressor(
            fc_layers  =cfg.num_fclayers,
            use_dropout=cfg.use_dropout, 
            drop_prob  =cfg.drop_prob, 
            use_ac_func=cfg.ac_func,
            num_param  =self.num_param,
            num_iters  =cfg.num_iter,
            max_batch_size=self.max_batch_size)

        # Load MANO hand model layer
        self.mano = MANO()
        self.cfg = cfg


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
        faces = torch.tensor(faces).cuda()
        # # Convert from m to mm
        vert *= 1000.0
        joint *= 1000.0

        if self.cfg.pred_scale:
            vert[:,:,:2] = vert[:,:,:2]*scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)
            joint[:,:,:2] = joint[:,:,:2]*scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)
            keypt = joint[:,:,:2] 
        else:
        # Project 3D joints to 2D image using weak perspective projection 
        # only consider x and y axis so does not rely on camera intrinsic
        # [bs,21,2] * [bs,1,1] + [bs,1,2]
            keypt = joint[:,:,:2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)

        # For STB dataset joint 0 is at palm center instead of wrist
        # Use half the distance between wrist and middle finger MCP as palm center (root joint)
        if self.stb_dataset:
            joint[:,0,:] = (joint[:,0,:] + joint[:,9,:])/2.0
        
        # if self.stb_dataset: # For publication to generate images for STB dataset
        # if not self.stb_dataset:
        vert  = vert  - joint[:,9,:].unsqueeze(1) # Make all vert relative to middle finger MCP
        joint = joint - joint[:,9,:].unsqueeze(1) # Make all joint relative to middle finger MCP
        keypt = keypt - keypt[:,9,:].unsqueeze(1)
            
        return keypt, joint, vert, ang, faces # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23]

    def get_theta_param(self, params):
        ang = params[:, 16:].contiguous()
        pose = self.mano.convert_ang_to_pose(ang)
        theta_param = torch.cat((params[:,:16], pose), dim=1)

        return theta_param


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
    # device = torch.device('cuda' if torch.cuda.is_available() and True else 'cpu')
    model = HMR() 
    # model.load_state_dict(torch.load('C:/Users/UVRLab/Desktop/sfGesture/model/hmr_model_freihand_auc.pth'))
    model.eval()

    # display_num_param(model) # 3.82 million for MobileNetV3 Small
    # print(model.state_dict().keys())
    # model.eval()

    bs = 10 # Batch size
    image = torch.randn(bs,1,224,224)
    print(image.shape)
    keypt, joint, vert, ang, faces, params = model(image)
    mano = MANO()
    lis = []

    for i in range(ang.shape[0]):
        li = mano.compute_ang_limit(ang[i])
        lis.append(li.detach())

    beta = params[:, 6:16]
    beta_norm = np.linalg.norm(beta.detach().numpy(), axis=1, ord=2) ** 2
    loss = np.asarray(lis) + beta_norm
    print(np.mean(loss))



    # print(torch.max(joint))

    # print('keypt', keypt.shape) # [bs,21,2]
    # print('joint', joint.shape) # [bs,21,3]
    # print('vert', vert.shape) # [bs,778,3]
    # print('ang', ang.shape, ang[0][0]) # [bs,23] 
    # print('faces', faces.shape) #(1538,3)#
    # print('params', len(params),params[0][3:6]) # 3 [bs,39] #vrec
