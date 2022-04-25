import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np
import os
import sys
sys.path.append('.')
sys.path.append('..')

from utils.utils_mpi_model import MANO
from utils.resnet import resnet152 
try:
    from efficientnet_pt.model import EfficientNet
except:
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from efficientnet_pt import EfficientNet
from utils.train_utils import normalize_image, compute_uv_from_integral
from net.hg_hm import Net_HM_HG

bases_num = 10 
pose_num = 30
mesh_num = 778
keypoints_num = 16

def rodrigues(r):       
    theta = torch.sqrt(torch.sum(torch.pow(r, 2),1))  

    def S(n_):   
        ns = torch.split(n_, 1, 1)     
        Sn_ = torch.cat([torch.zeros_like(ns[0]),-ns[2],ns[1],ns[2],torch.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],torch.zeros_like(ns[0])], 1)
        Sn_ = Sn_.view(-1, 3, 3)      
        return Sn_    

    n = r/(theta.view(-1, 1))   
    Sn = S(n) 

    #R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)
    
    I3 = Variable(torch.eye(3).unsqueeze(0).to(r.device))

    R = I3 + torch.sin(theta).view(-1, 1, 1)*Sn\
        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    Sr = S(r)
    theta2 = theta**2     
    R2 = I3 + (1.-theta2.view(-1,1,1)/6.)*Sr\
        + (.5-theta2.view(-1,1,1)/24.)*torch.matmul(Sr,Sr)
    
    idx = np.argwhere((theta<1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx,:,:] = R2[idx,:,:]

    return R,Sn

def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3                                                    
    pose_matrix, _ = rodrigues(poses[:,1:,:].contiguous().view(-1,3))
    #pose_matrix, _ = rodrigues(poses.view(-1,3))    
    pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(keypoints_num-1),axis=0)).to(device=poses.device))
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix

def rot_pose_beta_to_mesh(rots, poses, betas):
    #import pdb; pdb.set_trace()
    #dd = pickle.load(open('examples/data/MANO_RIGHT.pkl', 'rb'),encoding='latin1')
    MANO_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),'./model/MANO_RIGHT.pkl')
    dd = pickle.load(open(MANO_file, 'rb'),encoding='latin1')
    kintree_table = dd['kintree_table']
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])} 
    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}  

    devices = rots.device

    mesh_mu = Variable(torch.from_numpy(np.expand_dims(dd['v_template'], 0).astype(np.float32)).to(device=devices)) # zero mean
    mesh_pca = Variable(torch.from_numpy(np.expand_dims(dd['shapedirs'], 0).astype(np.float32)).to(device=devices))
    posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).to(device=devices))
    J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).to(device=devices))
    weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).to(device=devices))
    hands_components = Variable(torch.from_numpy(np.expand_dims(np.vstack(dd['hands_components'][:pose_num]), 0).astype(np.float32)).to(device=devices))
    hands_mean = Variable(torch.from_numpy(np.expand_dims(dd['hands_mean'], 0).astype(np.float32)).to(device=devices))
    root_rot = Variable(torch.FloatTensor([np.pi,0.,0.]).unsqueeze(0).to(device=devices))

    mesh_face = Variable(torch.from_numpy(np.expand_dims(dd['f'],0).astype(np.int16)).to(device=devices))
    
    #import pdb; pdb.set_trace()
    batch_size = rots.size(0)   

    mesh_face = mesh_face.repeat(batch_size, 1, 1)
    #import pdb; pdb.set_trace()
    poses = (hands_mean + torch.matmul(poses.unsqueeze(1), hands_components).squeeze(1)).view(batch_size,keypoints_num-1,3)
    # [b,15,3] [0:3]index [3:6]mid [6:9]pinky [9:12]ring [12:15]thumb

    #import pdb; pdb.set_trace()

    # for visualization
    #rots = torch.zeros_like(rots); rots[:,0]=np.pi/2


    #poses = torch.ones_like(poses)*1
    #poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)   
    poses = torch.cat((root_rot.repeat(batch_size,1).view(batch_size,1,3),poses),1) # [b,16,3]

    v_shaped =  (torch.matmul(betas.unsqueeze(1), 
                mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,bases_num,-1)).squeeze(1)    
                + mesh_mu.repeat(batch_size,1,1).view(batch_size, -1)).view(batch_size, mesh_num, 3)      
    
    pose_weights = get_poseweights(poses, batch_size)#[b,135]   
    
    v_posed = v_shaped + torch.matmul(posedirs.repeat(batch_size,1,1,1),
              (pose_weights.view(batch_size,1,(keypoints_num - 1)*9,1)).repeat(1,mesh_num,1,1)).squeeze(3)

    J_posed = torch.matmul(v_shaped.permute(0,2,1),J_regressor.repeat(batch_size,1,1).permute(0,2,1))
    J_posed = J_posed.permute(0, 2, 1)
    J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]
         
    pose = poses.permute(1, 0, 2)
    pose_split = torch.split(pose, 1, 0)
    #import pdb; pdb.set_trace()

    angle_matrix =[]
    for i in range(keypoints_num):
        out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
        angle_matrix.append(out)

    #with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

    with_zeros = lambda x:\
        torch.cat((x,   Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1).to(device=devices))  ),1)

    pack = lambda x: torch.cat((Variable(torch.zeros(batch_size,4,3).to(device=devices)),x),2) 

    results = {}
    results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size,3,1)),2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(torch.cat((angle_matrix[i],
                         (J_posed_split[i] - J_posed_split[parent[i]]).view(batch_size,3,1)),2)) 
        results[i] = torch.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []
         
    for i in range(len(results)):
        vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size,1).to(device=devices)) ),1)).view(batch_size,4,1)
        results2.append((results[i]-pack(torch.matmul(results[i], vec))).unsqueeze(0))    

    results = torch.cat(results2, 0)
    
    T = torch.matmul(results.permute(1,2,3,0), weights.repeat(batch_size,1,1).permute(0,2,1).unsqueeze(1).repeat(1,4,1,1))
    Ts = torch.split(T, 1, 2)
    rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size,mesh_num,1).to(device=devices)) ), 2)  
    rest_shape_hs = torch.split(rest_shape_h, 1, 2)

    v = Ts[0].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, mesh_num)\
        + Ts[1].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, mesh_num)\
        + Ts[2].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, mesh_num)\
        + Ts[3].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, mesh_num)
   
    #v = v.permute(0,2,1)[:,:,:3] 
    Rots = rodrigues(rots)[0]
    #import pdb; pdb.set_trace()
    Jtr = []

    for j_id in range(len(results_global)):
        Jtr.append(results_global[j_id][:,:3,3:4])

    # Add finger tips from mesh to joint list    
    '''
    Jtr.insert(4,v[:,:3,333].unsqueeze(2))
    Jtr.insert(8,v[:,:3,444].unsqueeze(2))
    Jtr.insert(12,v[:,:3,672].unsqueeze(2))
    Jtr.insert(16,v[:,:3,555].unsqueeze(2))
    Jtr.insert(20,v[:,:3,745].unsqueeze(2)) 
    '''
    # For FreiHand
    Jtr.insert(4,v[:,:3,320].unsqueeze(2))
    Jtr.insert(8,v[:,:3,443].unsqueeze(2))
    Jtr.insert(12,v[:,:3,672].unsqueeze(2))
    Jtr.insert(16,v[:,:3,555].unsqueeze(2))
    Jtr.insert(20,v[:,:3,744].unsqueeze(2))      
     
    Jtr = torch.cat(Jtr, 2) #.permute(0,2,1)
    
    v = torch.matmul(Rots,v[:,:3,:]).permute(0,2,1) #.contiguous().view(batch_size,-1)
    Jtr = torch.matmul(Rots,Jtr).permute(0,2,1) #.contiguous().view(batch_size,-1)
    
    #return torch.cat((Jtr,v), 1)
    return torch.cat((Jtr,v), 1), mesh_face, poses

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class Encoder(nn.Module):
    def __init__(self,version='b3'):
        super(Encoder, self).__init__()
        self.version = version
        if self.version == 'b3':
            #self.encoder = EfficientNet.from_pretrained('efficientnet-b3')
            self.encoder = EfficientNet.from_name('efficientnet-b3')
            # b3 [1536,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)
        '''
        elif self.version == 'b5':
            self.encoder = EfficientNet.from_pretrained('efficientnet-b5')
            # b5 [2048,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)
        '''
    def forward(self, x):
        features, low_features = self.encoder.extract_features(x)#[B,1536,7,7] [B,32,56,56]
        features = self.pool(features)
        features = features.view(features.shape[0],-1)##[B,1536]
        return features, low_features

class MyHandDecoder(nn.Module):
    def __init__(self,inp_neurons=1536,use_mean_shape=False):
        super(MyHandDecoder, self).__init__()
        self.hand_decode = MyPoseHand(inp_neurons=inp_neurons,use_mean_shape = use_mean_shape)
        if use_mean_shape:
            print("use mean MANO shape")
        else:
            print("do not use mean MANO shape")
        #self.hand_faces = self.hand_decode.mano_branch.faces

    def forward(self, features):
        #sides = torch.zeros(features.shape[0],1)
        #verts, faces, joints = self.hand_decode(features, Ks)
        '''
        joints, verts, faces, theta, beta = self.hand_decode(features)
        return joints, verts, faces, theta, beta
        '''
        joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses = self.hand_decode(features)
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses

class MyPoseHand(nn.Module):
    def __init__(
        self,
        ncomps=6,
        inp_neurons=1536,
        use_pca=True,
        dropout=0,
        use_mean_shape = False,
        ):
        super(MyPoseHand, self).__init__()
        self.use_mean_shape = use_mean_shape
        #import pdb;pdb.set_trace()
        # Base layers
        base_layers = []
        base_layers.append(nn.Linear(inp_neurons, 1024))
        base_layers.append(nn.BatchNorm1d(1024))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Linear(1024, 512))
        base_layers.append(nn.BatchNorm1d(512))
        base_layers.append(nn.ReLU())
        self.base_layers = nn.Sequential(*base_layers)

        # Pose Layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 30))#6
        self.pose_reg = nn.Sequential(*layers)

        # Shape Layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 10))
        self.shape_reg = nn.Sequential(*layers)

        # Trans layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 3))
        self.trans_reg = nn.Sequential(*layers)

        # rot layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 3))
        self.rot_reg = nn.Sequential(*layers)

        # scale layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 1))
        #layers.append(nn.ReLU())
        self.scale_reg = nn.Sequential(*layers)

        self.init_weights()
        #self.mean = torch.zeros()
        #self.mean = Variable(torch.FloatTensor([400,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]).cuda())
    def init_weights(self):
        #import pdb; pdb.set_trace()
        #len_scale_reg = len(self.scale_reg)
        '''
        for m in self.scale_reg:
            #import pdb; pdb.set_trace()
            if hasattr(m, 'weight'):#remove ReLU 
                normal_init(m, std=0.1,bias=0.95)
        '''
        normal_init(self.scale_reg[0],std=0.001)
        normal_init(self.scale_reg[2],std=0.001)
        normal_init(self.scale_reg[3],std=0.001,bias=0.95)

        normal_init(self.trans_reg[0],std=0.001)
        normal_init(self.trans_reg[2],std=0.001)
        normal_init(self.trans_reg[3],std=0.001)
        nn.init.constant_(self.trans_reg[3].bias[2],0.65)
        

    def forward(self, features):
        #import pdb; pdb.set_trace()
        base_features = self.base_layers(features)
        theta = self.pose_reg(base_features)#pose
        beta = self.shape_reg(base_features)#shape
        scale = self.scale_reg(base_features)
        trans = self.trans_reg(base_features)
        rot = self.rot_reg(base_features)
        '''
        mano = self.mano_regressor(features)
        mano = mano + mano.mul(self.mean.repeat(mano.shape[0],1).to(device=mano.device))
        scale = mano[:,0]
        trans = mano[:,1:4]
        rot = mano[:,4:7]
        theta = mano[:,7:13]#pose
        beta = mano[:,13:]
        '''
        if self.use_mean_shape:
            beta = torch.zeros_like(beta).to(beta.device)
        # try to set theta as zero tensor
        #theta = torch.zeros_like(theta)#
        #import pdb; pdb.set_trace()
        jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)#rotation pose shape
        #import pdb; pdb.set_trace()
        jv_ts = trans.unsqueeze(1) + torch.abs(scale.unsqueeze(2)) * jv[:,:,:]
        #jv_ts = jv_ts.view(x.size(0),-1) 
        joints = jv_ts[:,0:21]
        verts = jv_ts[:,21:]
        #import pdb; pdb.set_trace()
        #joints,  verts, faces = pose_hand(mano, K)
        #joints,  verts, faces = None, None, None
        #return joints, verts, faces, theta, beta
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses

class RGB2HM(nn.Module):
    def __init__(self):
        super(RGB2HM, self).__init__()
        num_joints = 21
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=2,
                                num_modules=2,
                                num_feats=256)
    def forward(self, images):
        # 1. Heat-map estimation
        est_hm_list, encoding = self.net_hm(images)
        return est_hm_list, encoding

class Model(nn.Module):
    def __init__(self, train_requires=['joints', 'heatmaps']):
        super(Model, self).__init__()
        
        # 2D hand estimation
        if 'heatmaps' in train_requires:
            self.rgb2hm = RGB2HM()

        # 3D hand estimation
        if "joints" in train_requires:
            self.regress_mode = 'mano'
            self.use_mean_shape = True
            if self.regress_mode == 'mano':# efficient-b3
                self.encoder = Encoder()
                self.dim_in = 1536
                self.hand_decoder = MyHandDecoder(inp_neurons=self.dim_in, use_mean_shape = self.use_mean_shape)
        else:
            self.regress_mode = None

    def predict_singleview(self, images, requires):
        vertices, faces, joints, shape, pose, trans, segm_out, textures, lights = None, None, None, None, None, None, None, None, None
        re_images, re_sil, re_img, re_depth, gt_depth = None, None, None, None, None
        pca_text, face_textures = None, None
        output = {}
        # 1. Heat-map estimation
        #end = time.time()

        if self.regress_mode == 'mano' or self.regress_mode == 'mano1':
            images = normalize_image(images)
            features, low_features = self.encoder(images)#[b,1536]

            if 'heatmaps' in requires:
                images_hm = images
                if images_hm.shape[3] != 256:
                    pad = nn.ZeroPad2d(padding=(0,32,0,32))
                    images_hm = pad(images_hm) #[bs, 3, 256, 256]
                hm_list, encoding = self.rgb2hm(images_hm)

                hm_keypt_list = []
                for hm in hm_list:
                    hm_keypt = compute_uv_from_integral(hm, images_hm.shape[2:4]) #[bs, 21, 3]
                    hm_keypt_list.append(hm_keypt)

                output['hm_list'] = hm_list
                output['hm_keypt_list'] = hm_keypt_list
                output['hm_2d_keypt_list'] = [hm_keypt[:,:,:2] for hm_keypt in hm_keypt_list]

            if 'joints' in requires or 'verts' in requires:
                #joints, vertices, faces, pose, shape = self.hand_decoder(features)
                joints, vertices, faces, pose, shape, scale, trans, rot, tsa_poses  = self.hand_decoder(features)

        output['joints'] = joints
        output['vertices'] = vertices
        output['pose'] = pose
        output['shape'] = shape
        output['scale'] = scale
        output['trans'] = trans
        output['rot'] = rot
        output['tsa_poses'] = tsa_poses
        output['faces'] = faces

        '''
        #del features
        #import pdb; pdb.set_trace()
        # 4. Render image
        faces = faces.type(torch.int32)
        if self.render_choice == 'NR':
            # use neural renderer
            #I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
            #Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).to(Ks.device)
            # create textures
            if textures is None:
                texture_size = 1
                textures = torch.ones(faces.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(vertices.device)
            
            self.renderer_NR.R = torch.unsqueeze(torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float(),0).repeat(Ks.shape[0],1,1).to(vertices.device)
            self.renderer_NR.t = torch.unsqueeze(torch.tensor([[0,0,0]]).float(),0).repeat(Ks.shape[0],1,1).to(vertices.device)
            self.renderer_NR.K = Ks[:,:,:3].to(vertices.device)
            self.renderer_NR.dist_coeffs = self.renderer_NR.dist_coeffs.to(vertices.device)
            #import pdb; pdb.set_trace()
            
            face_textures = textures.view(textures.shape[0],textures.shape[1],1,1,1,3)
            
            re_img,re_depth,re_sil = self.renderer_NR(vertices, faces, torch.tanh(face_textures), mode=None)

            re_depth = re_depth * (re_depth < 1).float()#set 100 into 0

            #import pdb; pdb.set_trace()
            if self.get_gt_depth and gt_verts is not None:
                gt_depth = self.renderer_NR(gt_verts, faces, mode='depth')
                gt_depth = gt_depth * (gt_depth < 1).float()#set 100 into 0
            #import pdb; pdb.set_trace()
        
        output['faces'] = faces
        output['re_sil'] = re_sil
        output['re_img'] = re_img
        output['re_depth'] = re_depth
        output['gt_depth'] = gt_depth
        '''
        
        return output
    def forward(self, images=None, task='train', requires=['joints', 'heatmaps']):
        if task == 'train' or task == 'hm_train':
            return self.predict_singleview(images, requires)


if __name__ == "__main__":
    model = Model()
    print(model)
