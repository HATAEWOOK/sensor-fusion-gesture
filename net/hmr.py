"""
Network for hand rotation using mano hand model
Input : RGB(? x ?) temp 224*224
Output : rvec(3)
"""
import encodings
from pickle import NONE
import numpy as np
import torch
import torch.nn as nn
import sys
import imageio
from hmr_s2 import RGB2HM
# sys.path.append("C:\\Users\\UVRLab\\Desktop\\sfGesture")
sys.path.append('.')
sys.path.append('..')

from utils.linear_model import LinearModel

from utils.utils_mpi_model import MANO #temp
import utils.utils_mobilenet_v3 as utils_mobilenet_v3 #tmep
from utils.resnet import resnet50
from utils.train_utils import normalize_image, compute_uv_from_integral
from net.hg_hm import Net_HM_HG

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
        input : output of encoder which has 2048 features
        return: list of params, 
        """
        params = []
        bs = inputs.shape[0] #batch size
        param = self.mean_param[:bs, :] #[bs, num_param]

        for _ in range(self.num_iters):
            total = torch.cat([inputs, param], dim=1)
            param = param + self.fc_blocks(total)
            params.append(param)

        return params

class HMR(nn.Module):
    def __init__(self, cfg):
        super(HMR, self).__init__()

        # Number of parameters to be regressed
        # Scaling:1, Translation:2, Global rotation :3, Beta:10, Joint angle:23
        # self.num_param = 39 # 1 + 2 + 3 + 10 + 23
        self.num_param = 61 # 1 + 2 + 3 + 10 + 45
        self.max_batch_size = 300
        self.stb_dataset = False

        # Load encoder 
        # self.encoder = utils_mobilenet_v3.mobilenetv3_small() # MobileNetV3
        # num_features = 576
        if cfg.pretrained:
            self.encoder = resnet50(pretrained=True)
        else:
            self.encoder = resnet50()
        
        # self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = Identity()

        # num_features = 2048
        
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
        self.rgb2hm = RGB2HM()


    def compute_results(self, param):
        # From the input parameters [bs, num_param] 
        # Compute the resulting 2D marker location
        scale = param[:, 0].contiguous()    # [bs]    Scaling 
        trans = param[:, 1:3].contiguous()  # [bs,2]  Translation in x and y only
        rvec  = param[:, 3:6].contiguous()  # [bs,3]  Global rotation vector
        beta  = param[:, 6:16].contiguous() # [bs,10] Shape parameters
        ang   = param[:, 16:].contiguous()  # [bs,23] Angle parameters

        pose = self.mano.convert_pca_to_pose(ang)
        # rvec = torch.tanh(rvec)*np.pi
        vert, joint = self.mano(beta, pose, rvec)
        faces = self.mano.F
        faces = torch.tensor(faces).cuda()
        # # Convert from m to mm
        vert *= 1000.0
        joint *= 1000.0

        # vert  = vert  - joint[:,9,:].unsqueeze(1) # Make all vert relative to middle finger MCP
        # joint_normal = joint - joint[:,9,:].unsqueeze(1) # Make all joint relative to middle finger MCP

        # norm_scale = torch.norm(joint[:,10,:] - joint[:, 9,:], dim = 1) #[bs]
        # joint_normal = joint_normal / norm_scale.unsqueeze(1).unsqueeze(2)

        # Project 3D joints to 2D image using weak perspective projection 
        # only consider x and y axis so does not rely on camera intrinsic
        # [bs,21,2] * [bs,1,1] + [bs,1,2]
        keypt = joint[:,:,:2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)
        # print("===============")
        # print(torch.max(keypt[0]))
        # print(torch.min(keypt[0]))

        # For STB dataset joint 0 is at palm center instead of wrist
        # Use half the distance between wrist and middle finger MCP as palm center (root joint)
        if self.stb_dataset:
            joint[:,0,:] = (joint[:,0,:] + joint[:,9,:])/2.0
        
        # if self.stb_dataset: # For publication to generate images for STB dataset
        # if not self.stb_dataset:
            
        return keypt, joint, vert, ang, pose, faces # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23]

    def get_theta_param(self, params):
        ang = params[:, 16:].contiguous()
        pose = self.mano.convert_ang_to_pose(ang)
        theta_param = torch.cat((params[:,:16], pose.view(-1,45)), dim=1)

        return theta_param

    def rendering(self, vert, faces, Ks):
        import neural_renderer as nr
        faces = faces.type(torch.int32)
        faces = faces.repeat(Ks.shape[0], 1, 1)
        texture_size = 1
        textures = torch.ones(faces.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(vert.device)
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        Ps = torch.bmm(Ks, Is)
        face_textures = textures.view(textures.shape[0],textures.shape[1],1,1,1,3)
        renderer = nr.Renderer(image_size = 224, camera_mode='projection', orig_size=224)
        
        renderer.R = torch.unsqueeze(torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float(),0).repeat(Ps.shape[0],1,1).to(vert.device)
        renderer.t = torch.unsqueeze(torch.tensor([[0,0,0]]).float(),0).repeat(Ps.shape[0],1,1).to(vert.device)
        renderer.K = Ps[:,:,:3].to(vert.device)
        renderer.dist_coeffs = renderer.dist_coeffs.to(vert.device)

        re_img, re_depth, re_sil = renderer(vert, faces, torch.tanh(face_textures), mode=None)

        return re_img, re_depth, re_sil


    def forward(self, inputs, evaluation=True):
        inputs = normalize_image(inputs) #[10,3,224,224]
        features = self.encoder(inputs)
        params   = self.regressor(features.squeeze())
        output = {}

        #heatmap 2d 
        inputs_hm = inputs
        if inputs_hm.shape[3] != 256:
            pad = nn.ZeroPad2d(padding=(0,32,0,32))
            inputs_hm = pad(inputs_hm)
        hm_list, encoding = self.rgb2hm(inputs_hm) #{[10,21,64,64]}, {[10,256,64,64]}
        hm_keypt_list = []
        for hm in hm_list:
            hm_keypt = compute_uv_from_integral(hm, inputs_hm.shape[2:4]) #[bs, 21, 3]
            hm_keypt_list.append(hm_keypt)

        output['hm_list'] = hm_list
        output['hm_keypt_list'] = hm_keypt_list
        output['hm_2d_keypt_list'] = [hm_keypt[:,:,:2] for hm_keypt in hm_keypt_list]

        if evaluation:
            # Only return final param
            keypt, joint, vert, ang, pose, faces = self.compute_results(params[-1])
            output['keypt'] = keypt
            output['joint'] = joint
            output['vert'] = vert
            output['ang'] = ang
            output['pose'] = pose
            output['faces'] = faces
            output['param'] = params[-1]
            # re_img, re_depth, re_sil = self.rendering(vert, faces, Ks)
            # if save_path is not None:
            #     writer = imageio.get_writer(save_path, mode="i")
            #     re_img_tmp = re_img.detach().cpu().numpy()[0].transpose((1,2,0))
            #     writer.append_data((255*re_img_tmp).astype(np.uint8))
            #     writer.close()
            #     del re_img_tmp
            return output

        else: #training
            keypt, joint, vert, ang, pose, faces = self.compute_results(params[-1])
            output['keypt'] = keypt
            output['joint'] = joint
            output['vert'] = vert
            output['ang'] = ang
            output['pose'] = pose
            output['faces'] = faces
            output['param'] = params[-1]

            return output

class HMR_tuning(nn.Module):
    def __init__(self, cfg):
        super(HMR_tuning, self).__init__()

        # Number of parameters to be regressed
        # Scaling:1, Translation:2, Global rotation :3, Beta:10, Joint angle:23
        # self.num_param = 39 # 1 + 2 + 3 + 10 + 23
        self.num_param = 61 # 1 + 2 + 3 + 10 + 45
        self.max_batch_size = 300
        self.stb_dataset = False

        # Load encoder 
        # self.encoder = utils_mobilenet_v3.mobilenetv3_small() # MobileNetV3
        # num_features = 576
        if cfg["pretrained"]:
            self.encoder = resnet50(pretrained=True)
        else:
            self.encoder = resnet50()
        
        # self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = Identity()

        # num_features = 2048
        
        # Load iterative regressor
        self.regressor = Regressor(
            fc_layers  =cfg["num_fclayers"],
            use_dropout=cfg["use_dropout"], 
            drop_prob  =cfg["drop_prob"],
            use_ac_func=cfg["ac_func"],
            num_param  =self.num_param,
            num_iters  =cfg["num_iter"],
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

        pose = self.mano.convert_pca_to_pose(ang)
        # rvec = torch.tanh(rvec)*np.pi
        vert, joint = self.mano(beta, pose, rvec)
        faces = self.mano.F
        faces = torch.tensor(faces).cuda()
        # # Convert from m to mm
        vert *= 1000.0
        joint *= 1000.0

        # vert  = vert  - joint[:,9,:].unsqueeze(1) # Make all vert relative to middle finger MCP
        # joint_normal = joint - joint[:,9,:].unsqueeze(1) # Make all joint relative to middle finger MCP

        # norm_scale = torch.norm(joint[:,10,:] - joint[:, 9,:], dim = 1) #[bs]
        # joint_normal = joint_normal / norm_scale.unsqueeze(1).unsqueeze(2)

        # Project 3D joints to 2D image using weak perspective projection 
        # only consider x and y axis so does not rely on camera intrinsic
        # [bs,21,2] * [bs,1,1] + [bs,1,2]
        keypt = joint[:,:,:2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)
        # print("===============")
        # print(torch.max(keypt[0]))
        # print(torch.min(keypt[0]))

        # For STB dataset joint 0 is at palm center instead of wrist
        # Use half the distance between wrist and middle finger MCP as palm center (root joint)
        if self.stb_dataset:
            joint[:,0,:] = (joint[:,0,:] + joint[:,9,:])/2.0
        
        # if self.stb_dataset: # For publication to generate images for STB dataset
        # if not self.stb_dataset:
            
        return keypt, joint, vert, ang, pose, faces # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23]

    def get_theta_param(self, params):
        ang = params[:, 16:].contiguous()
        pose = self.mano.convert_ang_to_pose(ang)
        theta_param = torch.cat((params[:,:16], pose.view(-1,45)), dim=1)

        return theta_param

    def rendering(self, vert, faces, Ks):
        import neural_renderer as nr
        faces = faces.type(torch.int32)
        faces = faces.repeat(Ks.shape[0], 1, 1)
        texture_size = 1
        textures = torch.ones(faces.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(vert.device)
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        Ps = torch.bmm(Ks, Is)
        face_textures = textures.view(textures.shape[0],textures.shape[1],1,1,1,3)
        renderer = nr.Renderer(image_size = 224, camera_mode='projection', orig_size=224)
        
        renderer.R = torch.unsqueeze(torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float(),0).repeat(Ps.shape[0],1,1).to(vert.device)
        renderer.t = torch.unsqueeze(torch.tensor([[0,0,0]]).float(),0).repeat(Ps.shape[0],1,1).to(vert.device)
        renderer.K = Ps[:,:,:3].to(vert.device)
        renderer.dist_coeffs = renderer.dist_coeffs.to(vert.device)

        re_img, re_depth, re_sil = renderer(vert, faces, torch.tanh(face_textures), mode=None)

        return re_img, re_depth, re_sil


    def forward(self, inputs, evaluation=True, get_feature=False):
        inputs = normalize_image(inputs)
        features = self.encoder(inputs)
        params   = self.regressor(features.squeeze())

        if evaluation:
            # Only return final param
            keypt, joint, vert, ang, pose, faces = self.compute_results(params[-1])
            # re_img, re_depth, re_sil = self.rendering(vert, faces, Ks)
            # if save_path is not None:
            #     writer = imageio.get_writer(save_path, mode="i")
            #     re_img_tmp = re_img.detach().cpu().numpy()[0].transpose((1,2,0))
            #     writer.append_data((255*re_img_tmp).astype(np.uint8))
            #     writer.close()
            #     del re_img_tmp
            if get_feature:
                return keypt, joint, vert, ang, pose, faces, params[-1], features
            else:
                return keypt, joint, vert, ang, pose, faces, params[-1]
            # results  = []
            # for param in params:
            #     results.append(self.compute_results(param))
        else:
            keypt, joint, vert, ang, pose, faces = self.compute_results(params[-1])

            return keypt, joint, vert, ang, pose, faces, params[-1] # Return the list of params at each iteration
###############################################################################
### Simple example to test the program                                      ###
###############################################################################
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() and True else 'cpu')
    rgb2hm = RGB2HM()
    input = torch.randn(10, 3, 224, 224).cuda()
    pad = nn.ZeroPad2d(padding=(0,32,0,32))
    input = pad(input)
    rgb2hm.cuda()
    rgb2hm.eval()
    hm_list, encoding = rgb2hm(input)
    print(len(hm_list))
    hm_keypt_list = []
    for hm in hm_list:
        hm_keypt = compute_uv_from_integral(hm, input.shape[2:4]) #[bs, 21, 3]
        hm_keypt_list.append(hm_keypt)

    print(hm_keypt_list[0].shape)
    print(hm_keypt_list[1].shape)