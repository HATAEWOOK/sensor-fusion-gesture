"""
Trainting for get rotation of hand from mano hand model
Input : RGB(? x ?)
Output : rvec
"""
import sys
sys.path.append('.')
sys.path.append('..')
# sys.path.append("C:\\Users\\UVRLab\\Desktop\\sfGesture")
# sys.path.append('/root/sensor-fusion-gesture')
import os
import numpy as np
from datetime import datetime
import random
import time
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import neural_renderer as nr
import open3d as o3d

from datasetloader.data_loader_MSRAHT import get_dataset
from net.hmr import HMR
from utils.train_utils import mklogger, AverageMeter, Mano2depth, Data_preprocess, ResultGif, proj_func, set_vis, save_image, regularizer_loss, shape_loss, direction_loss, normalize, proj_func, orthographic_proj_withz
from utils.config_parser import Config
from net.hmr_s2 import Model

class Trainer:
    def __init__(self, cfg, model, vis):
        # Initialize randoms seeds
        torch.cuda.manual_seed_all(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        random.seed(cfg.manual_seed)
        torch.backends.cudnn.benchmark = False

        starttime = datetime.now().replace(microsecond=0)

        #checkpoint dir
        os.makedirs(cfg.ckp_dir, exist_ok=True)
        self.save_path = os.path.join(cfg.ckp_dir,'results%d'%cfg.config_num)
        os.makedirs(self.save_path, exist_ok=True)
        #===
        # os.makedirs(os.path.join(self.save_path, 'pred'), exist_ok=True)
        # os.makedirs(os.path.join(self.save_path, 'target'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'img'), exist_ok=True)
        #===
        log_path = os.path.join(self.save_path, 'logs')
        os.makedirs(log_path, exist_ok=True)
        logger = mklogger(log_path).info
        self.logger = logger
        summary_logdir = os.path.join(self.save_path, 'summaries')
        # self.swriter = SummaryWriter()
        logger('[%s] - Started training, experiment code %s' % (cfg.expr_ID, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)
        logger('Base dataset_dir is %s' % cfg.dataset_dir)
        self.try_num = cfg.try_num


        self.model = model
        if cfg.pretrained_net:
            self.load_model(model, cfg.pretrained_net)
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize optimizer
        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
            )
        elif cfg.optimizer == "adam_amsgrad":
            optimizer = torch.optim.Adam(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), amsgrad=True
            )
        elif cfg.optimizer == "rms":
            optimizer = torch.optim.RMSprop(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model_params,
                lr=cfg.lr,
                # momentum=cfg.momentum,
                # weight_decay=cfg.weight_decay,
            )
        self.optimizer = optimizer

        # cuda 
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if use_cuda else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = torch.cuda.device_count() if cfg.use_multigpu else 1
        if use_cuda:
            logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        
        if cfg.use_multigpu:
            self.model = nn.DataParallel(self.model)
            logger("Training on Multiple GPU's")

        self.model = self.model.to(self.device)
        self.load_data(cfg)

        # load learning rate
        for group in optimizer.param_groups:
            group['lr'] = cfg.lr
            group['initial_lr'] = cfg.lr

        if cfg.lr_decay_gamma:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, cfg.lr_decay_step, gamma = cfg.lr_decay_gamma
                )

        if cfg.lr_reduce:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, 
            )

        # loss
        self.smoothl1_criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')

        # etc
        self.cfg = cfg
        self.start_epoch = 0
        self.vis = vis
        self.max_norm = 5

    def save_model(self):
        torch.save(self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) 
                    else self.model.state_dict(), self.cfg.best_model)

    def load_model(self, model, pretrained):
        if pretrained == "/root/Pretrained/S2HAND/texturehand_freihand.t7":
            state = torch.load(pretrained)
            if 'encoder' in state.keys() and hasattr(model, 'encoder'):
                model.encoder.load_state_dict(state['encoder'])
                print("Encoder state loaded")
            if'decoder' in state.keys() and hasattr(model, 'hand_decoder'):
                model.hand_decoder.load_state_dict(state['decoder'])
                print("Decoder state loaded")
        else:
            model.load_state_dict(torch.load(pretrained))

    def load_data(self, cfg):
        '''
        self.train_dataloader / self.val_dataloader ('depth','processed', 'cropped', 'trans')
        queries
        get_dataset(dat_name, set_name, base_path, queries, use_cache=True, train=False, split=None)
        ToDo : 1 dataset -> train / val split 

        return 0
        '''
        kwargs_train = {
            'num_workers' : cfg.num_workers,
            'batch_size' : cfg.train_batch_size,
            'shuffle' : True,
            'drop_last' : True,
        }

        kwargs_test = {
            'num_workers' : cfg.num_workers,
            'batch_size' : cfg.test_batch_size,
            'shuffle' : True,
            'drop_last' : True,
        }

        dat_name = cfg.dataset
        dat_dir = cfg.dataset_dir
        # train_queries = ['image', 'open_j2d', 'Ks', 'j3d', 'verts', 'mano']
        train_queries = ['trans_img', 'trans_open_j2d', 'trans_Ks', 'trans_j3d', 'trans_verts', 'mano']
        # evaluation_queries = ['image', 'Ks', 'open_j2d']
        train_dataset = get_dataset(dat_name, dat_dir, queries=train_queries, set_name='training')
        # test_dataset = get_dataset(dat_name, dat_dir, queries=evaluation_queries, set_name='evaluation')
        # train_dataset, _ = torch.utils.data.random_split(train_dataset, [1280, len(train_dataset) - 1280])
        # test_dataset, _ = torch.utils.data.random_split(test_dataset, [160, len(test_dataset) - 160])
        split = int(len(train_dataset)*0.8)
        train_split, valid_split = torch.utils.data.random_split(train_dataset, [split, len(train_dataset) - split])
        # test_split, _ = torch.utils.data.random_split(test_dataset, [160, len(test_dataset) - 160])

        # self.test_dataloader = DataLoader(test_dataset, **kwargs_test)
        self.train_dataloader = DataLoader(train_split, **kwargs_train)
        self.valid_dataloader = DataLoader(valid_split, **kwargs_test)
        print("Data size : ",len(self.train_dataloader), len(self.valid_dataloader))


    def train(self):
        avg_meter = AverageMeter()
        self.model.train()
        ckpt = 10
        t = time.time()
        self.start_epoch += 1

        for b_idx, (sample) in enumerate(self.train_dataloader):
            target_img = sample['trans_img'].to(self.device) #[bs, 3, 224, 224]
            target_joint = sample['trans_j3d'].to(self.device) #[bs,21,3]
            # target_open_j2d = sample['trans_open_j2d'].to(self.device) #[bs, 21, 2]
            # target_open_j2d_con = sample['open_j2d_con'].to(self.device)
            # target_mano = sample['mano'].to(self.device) #[bs, 1, 61]
            Ks = sample['trans_Ks'].to(self.device) #[bs, 3, 3]

            output = self.model(target_img)
            # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [bs, 15, 3], [1538,3], [bs, 61]

            keypt = output['keypt']
            joint = output['joint']
            vert = output['vert']
            ang = output['ang'] 
            pose = output['pose'] 
            faces = output['faces'] 
            params = output['param']

            hm_2d_keypt_list = output['hm_2d_keypt_list']
            hm_2d_keypt = hm_2d_keypt_list[1] #[bs, 21, 2]

            beta = params[:,6:16].contiguous()

            target_joint_norm = normalize(target_joint)
            joint_norm = normalize(joint)

            target_keypt = proj_func(target_joint, Ks)
            target_keypt_norm = normalize(target_keypt, mode='keypt')
            keypt_norm = normalize(keypt, mode='keypt')

            # if (b_idx+1) % ckpt == 0 and self.start_epoch % 10 == 0:
            #     vis_path =  os.path.join(self.save_path, 'E%d_%d_pred_3d.png'%(self.start_epoch, b_idx+1))
            #     m2d = Mano2depth(vert, faces, joint)
            #     m2d.img_save(Ks, self.vis, path=vis_path)
            # else:
            #     vis_path = None

            j3d_loss = self.mse_criterion(joint_norm.to(self.device), target_joint_norm.squeeze())
            j2d_loss = self.smoothl1_criterion(keypt_norm.to(self.device), target_keypt_norm.squeeze())
            # scale_loss = self.smoothl1_criterion(params[:, 0].to(self.device), target_mano.squeeze()[:,-6])
            # open_j2d_loss = self.joint_criterion(keypt.to(self.device), target_open_j2d.squeeze())
            reg_loss = regularizer_loss(pose)
            beta_loss = self.mse_criterion(beta, torch.zeros_like(beta).to(self.device))
            direc_loss = direction_loss(target_joint_norm, joint)

            loss = j3d_loss*self.cfg.j3d_loss_weight + \
                j2d_loss*self.cfg.j2d_loss_weight + \
                reg_loss*self.cfg.reg_loss_weight + \
                beta_loss*self.cfg.shape_loss_weight + \
                direc_loss*self.cfg.direc_loss_weight
                # scale_loss*self.cfg.scale_loss_weight

            train_loss_dict = {
                'j3d_loss':j3d_loss,
                'j2d_loss':j2d_loss,
                'reg_loss':reg_loss,
                'shape_loss':beta_loss,
                'direc_loss':direc_loss,
                'total_loss':loss,
            }
            # loss.requires_grad_(True)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            
            avg_meter.update(loss.detach().item())

            if (b_idx+1) % ckpt == 0:
                term = time.time() - t
                self.logger('Step : %s/%s' % (b_idx+1, len(self.train_dataloader)))
                self.logger('Train loss : %.5f' % avg_meter.avg)
                self.logger('time : %.5f' % term)
                self.logger("[Loss] j3d_loss : %.5f, j2d_loss : %.5f, reg_loss : %.5f, shape_loss : %.5f, direc_loss : %.5f, total_loss : %.5f" % (
                train_loss_dict['j3d_loss'],
                train_loss_dict['j2d_loss'],
                train_loss_dict['reg_loss'],
                train_loss_dict['shape_loss'],
                train_loss_dict['direc_loss'],
                train_loss_dict['total_loss']))

                # tar_joint = target_joint_norm[0].squeeze().detach().cpu().numpy()
                # pred_joint = joint[0].squeeze().detach().cpu().numpy()
                # tar_keypt = target_keypt_norm[0].squeeze().detach().cpu().numpy()
                # pred_keypt = keypt_norm[0].squeeze().detach().cpu().numpy()
                # tar_mano = target_mano[0].squeeze().detach().cpu().numpy()
                # pred_mano = torch.cat([params[0, 6:16], pose[0].view(45), params[0, 0:6]]).detach().cpu().numpy()

                # np.savetxt(os.path.join(self.save_path,'pred', 'E%d_%d_joint.txt'%(self.start_epoch, b_idx+1)), pred_joint)
                # np.savetxt(os.path.join(self.save_path,'target', 'E%d_%d_joint.txt'%(self.start_epoch, b_idx+1)), tar_joint)
                # np.savetxt(os.path.join(self.save_path, 'pred', 'E%d_%d_keypt.txt'%(self.start_epoch, b_idx+1)), pred_keypt)
                # np.savetxt(os.path.join(self.save_path, 'target', 'E%d_%d_keypt.txt'%(self.start_epoch, b_idx+1)), tar_keypt)
                # np.savetxt(os.path.join(self.save_path, 'pred', 'E%d_%d_mano.txt'%(self.start_epoch, b_idx+1)), pred_mano)
                # np.savetxt(os.path.join(self.save_path, 'target', 'E%d_%d_mano.txt'%(self.start_epoch, b_idx+1)), tar_mano)

        return avg_meter, train_loss_dict

    def valid(self):
        avg_meter = AverageMeter()
        pck_meter = AverageMeter()
        self.model.eval()
        ckpt = 500
        t = time.time()
        pck_threshold = 1.0

        for b_idx, (sample) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                target_img = sample['trans_img'].to(self.device) #[bs, 3, 224, 224]
                target_joint = sample['trans_j3d'].to(self.device) #[bs,21,3]
                target_open_j2d = sample['trans_open_j2d'].to(self.device) #[bs, 21, 2]
                target_mano = sample['mano'].to(self.device) #[bs, 1, 61]
                Ks = sample['trans_Ks'].to(self.device) #[bs, 3, 3]

                output = self.model(target_img)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [bs, 15, 3], [1538,3], [bs, 61]

                keypt = output['keypt']
                joint = output['joint']
                vert = output['vert']
                ang = output['ang'] 
                pose = output['pose'] 
                faces = output['faces'] 
                params = output['param']

                beta = params[:,6:16].contiguous()

                target_keypt = proj_func(target_joint, Ks)

                target_joint_norm = normalize(target_joint)
                joint_norm = normalize(joint)

                target_keypt_norm = normalize(target_keypt, mode='keypt')
                keypt_norm = normalize(keypt, mode='keypt')

                if (b_idx+1) % ckpt == 0 and self.start_epoch % 10 == 0:
                    vis_path_mesh =  os.path.join(self.save_path, 'img', 'E%d_%d_pred_mesh.png'%(self.start_epoch, b_idx+1))
                    vis_path_joint =  os.path.join(self.save_path, 'img', 'E%d_%d_pred_joint.png'%(self.start_epoch, b_idx+1))
                    m2d = Mano2depth(vert, faces, joint)
                    # m2d.joint_save(Ks, self.vis, path=vis_path_joint, radius=0.05)
                    m2d.mesh_save(Ks, self.vis, path=vis_path_mesh)
                    target = target_img[0].permute(1,2,0)
                    name = sample['idx'][0].item()
                    save_image(target, os.path.join(self.save_path, 'img', 'E%d_%d_target_%s.png'%(self.start_epoch, b_idx+1, name)))

                joint_distance = torch.norm(target_joint_norm - joint_norm, dim = 2) #[bs, 21] distance of each joints
                joint_index = torch.where(joint_distance > pck_threshold, torch.zeros_like(joint_distance), torch.ones_like(joint_distance))
                pck = torch.mean(joint_index)
                pck_meter.update(pck.item())

                j3d_loss = self.mse_criterion(joint_norm.to(self.device), target_joint_norm.squeeze())
                j2d_loss = self.smoothl1_criterion(keypt_norm.to(self.device), target_keypt_norm.squeeze())
                # scale_loss = self.smoothl1_criterion(params[:, 0].to(self.device), target_mano.squeeze()[:,-6])
                # open_j2d_loss = self.joint_criterion(keypt.to(self.device), target_open_j2d.squeeze())
                reg_loss = regularizer_loss(pose)
                beta_loss = self.mse_criterion(beta, torch.zeros_like(beta).to(self.device))
                direc_loss = direction_loss(target_joint_norm, joint)

                loss = j3d_loss*self.cfg.j3d_loss_weight + \
                    j2d_loss*self.cfg.j2d_loss_weight + \
                    reg_loss*self.cfg.reg_loss_weight + \
                    beta_loss*self.cfg.shape_loss_weight + \
                    direc_loss*self.cfg.direc_loss_weight
                    # scale_loss*self.cfg.scale_loss_weight

                valid_loss_dict = {
                    'j3d_loss':j3d_loss,
                    'j2d_loss':j2d_loss,
                    'reg_loss':reg_loss,
                    'shape_loss':beta_loss,
                    'direc_loss':direc_loss,
                    'total_loss':loss,
                }
                
                avg_meter.update(loss.item())

                if (b_idx+1) % ckpt == 0:
                    term = time.time() - t
                    self.logger('Step : %s/%s' % (b_idx+1, len(self.train_dataloader)))
                    self.logger('Train loss : %.5f' % avg_meter.avg)
                    self.logger('time : %.5f' % term)
                    self.logger("[Loss] j3d_loss : %.5f, j2d_loss : %.5f, reg_loss : %.5f, shape_loss : %.5f, direc_loss : %.5f, total_loss : %.5f" % (
                    valid_loss_dict['j3d_loss'],
                    valid_loss_dict['j2d_loss'],
                    valid_loss_dict['reg_loss'],
                    valid_loss_dict['shape_loss'],
                    valid_loss_dict['direc_loss'],
                    valid_loss_dict['total_loss']))
                    self.logger("[3D PCK] 3D PCK : %.3f"%pck_meter.avg)

        return avg_meter, valid_loss_dict, pck_meter


    def eval(self):
        avg_meter = AverageMeter()
        self.model.eval()
        ckpt = 10
        t = time.time()

        for b_idx, (sample) in enumerate(self.test_dataloader):
            with torch.no_grad():
                target_img = sample['image'].to(self.device) #[bs, 3, 224, 224]
                target_keypt = sample['open_j2d'].to(self.device)
                Ks = sample['Ks'].to(self.device)
                name = sample['idx']

                keypt, joint, vert, ang, pose, faces, params= self.model(target_img)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
                beta = params[:,6:16].contiguous()

                if (b_idx+1) % ckpt == 0 and self.start_epoch % 10 == 0:
                    vis_path_mesh =  os.path.join(self.save_path, 'E%d_%d_pred_mesh.png'%(self.start_epoch, b_idx+1))
                    vis_path_joint =  os.path.join(self.save_path, 'E%d_%d_pred_joint.png'%(self.start_epoch, b_idx+1))
                    m2d = Mano2depth(vert, faces, joint)
                    m2d.joint_save(Ks, self.vis, path=vis_path_joint)
                    m2d.mesh_save(Ks, self.vis, path=vis_path_mesh)
                else:
                    vis_path = None

                # target_keypt_norm = normalize(target_keypt, mode='keypt')
                # keypt_norm = normalize(keypt, mode='keypt')

                j2d_loss = self.smoothl1_criterion(keypt.to(self.device), target_keypt.squeeze())
                reg_loss = regularizer_loss(pose=pose)
                beta_loss = self.mse_criterion(beta, torch.zeros_like(beta).to(self.device))

                loss = j2d_loss*self.cfg.j2d_loss_weight + reg_loss*self.cfg.reg_loss_weight + beta_loss*self.cfg.shape_loss_weight

                eval_loss_dict = {
                    'j2d_loss':j2d_loss,
                    'reg_loss':reg_loss,
                    'shape_loss':beta_loss,
                    'total_loss':loss,
                }
                # loss.requires_grad_(True)
                avg_meter.update(loss.item())

                if (b_idx+1) % ckpt == 0:
                    term = time.time() - t
                    self.logger('Step : %s/%s' % (b_idx+1, len(self.test_dataloader)))
                    self.logger('Evaluation loss : %.5f' % avg_meter.avg)
                    self.logger('time : %.5f' % term)
                    self.logger("[Loss] j2d_loss : %.5f, reg_loss : %.5f, shape_loss : %.5f, total_loss : %.5f" % (
                    eval_loss_dict['j2d_loss'],
                    eval_loss_dict['reg_loss'],
                    eval_loss_dict['shape_loss'],
                    eval_loss_dict['total_loss']))

                if (b_idx+1) % ckpt == 0 and self.start_epoch % 10 == 0:
                    target = target_img[0].permute(1,2,0)
                    name = sample['idx'][0].item()
                    # pred_joint = joint[0].squeeze().cpu().numpy()
                    # pred_vert = vert[0].squeeze().cpu().numpy()
                    # pred_faces = faces.detach().cpu()
                    # Ks = Ks[0].detach().cpu().numpy()
                    save_image(target, os.path.join(self.save_path, 'E%d_%d_target_%s.png'%(self.start_epoch, b_idx+1, name)))
                    # np.savetxt(os.path.join(self.save_path, 'E%d_%d_joint.txt'%(self.start_epoch, b_idx+1)), pred_joint)
                    # np.savetxt(os.path.join(self.save_path, 'E%d_%d_vert.txt'%(self.start_epoch, b_idx+1)), pred_vert)
                    # np.savetxt(os.path.join(self.save_path, 'E%d_%d_faces.txt'%(self.start_epoch, b_idx+1)), pred_faces)
                    # np.savetxt(os.path.join(self.save_path, 'E%d_%d_Ks.txt'%(self.start_epoch, b_idx+1)), Ks)
                    del target
                    # del pred_joint
                    # del pred_vert
                    # del pred_faces

        return avg_meter, eval_loss_dict

    def fit(self, n_epochs = None):

        starttime = datetime.now().replace(microsecond=0)
        train_loss = {}
        j3d_loss = {}
        j2d_loss = {}
        reg_loss = {}
        shape_loss = {}
        direc_loss = {}
        valid_loss = {}
        pck = {}
        eval_loss = {}

        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr = np.inf
        best_loss = np.inf

        for epoch_num in range(1, n_epochs + 1):
            self.logger('===== starting Epoch # %03d' % epoch_num)

            train_avg_meter, train_loss_dict = self.train()
            self.logger("[Epoch: %d/%d] Train loss : %.5f" % (epoch_num, n_epochs, train_avg_meter.avg))
            self.logger("[Loss] j3d_loss : %.5f, j2d_loss : %.5f, reg_loss : %.5f, shape_loss : %.5f, direc_loss : %.5f, total_loss : %.5f" % (
                train_loss_dict['j3d_loss'],
                train_loss_dict['j2d_loss'],
                train_loss_dict['reg_loss'],
                train_loss_dict['shape_loss'],
                train_loss_dict['direc_loss'],
                train_loss_dict['total_loss']))

            if self.cfg.validation:
                valid_avg_meter, valid_loss_dict, pck_meter = self.valid()
                self.logger("[Epoch: %d/%d] Evaluation loss : %.5f" % (epoch_num, n_epochs, valid_avg_meter.avg))
                self.logger("[Loss] j3d_loss : %.5f, j2d_loss : %.5f, reg_loss : %.5f, shape_loss : %.5f, direc_loss : %.5f, total_loss : %.5f" % (
                    valid_loss_dict['j3d_loss'],
                    valid_loss_dict['j2d_loss'],
                    valid_loss_dict['reg_loss'],
                    valid_loss_dict['shape_loss'],
                    valid_loss_dict['direc_loss'],
                    valid_loss_dict['total_loss']))
                self.logger("[3D PCK] 3D PCK : %.3f"%pck_meter.avg)
                valid_loss[epoch_num] = valid_avg_meter.avg
                pck[epoch_num] = pck_meter.avg

            # if self.cfg.evaluation:            
            #     eval_avg_meter, eval_loss_dict = self.eval()
            #     self.logger("[Epoch: %d/%d] Evaluation loss : %.5f" % (epoch_num, n_epochs, eval_avg_meter.avg))
            #     self.logger("[Loss] j2d_loss : %.5f, reg_loss : %.5f, shape_loss : %.5f, total_loss : %.5f" % (
            #         eval_loss_dict['j2d_loss'],
            #         eval_loss_dict['reg_loss'],
            #         eval_loss_dict['shape_loss'],
            #         eval_loss_dict['total_loss']))
            #     eval_loss[epoch_num] = eval_avg_meter.avg
            
            train_loss[epoch_num] = train_avg_meter.avg
            j3d_loss[epoch_num] = train_loss_dict['j3d_loss'].detach().cpu()
            j2d_loss[epoch_num] = train_loss_dict['j2d_loss'].detach().cpu()
            reg_loss[epoch_num] = train_loss_dict['reg_loss'].detach().cpu()
            shape_loss[epoch_num] = train_loss_dict['shape_loss'].detach().cpu()
            direc_loss[epoch_num] = train_loss_dict['direc_loss'].detach().cpu()
                
            fig = plt.figure(figsize=(15, 15))
            fig.suptitle('%d epochs'%epoch_num, fontsize=16)
            ax1 = fig.add_subplot(331)
            ax2 = fig.add_subplot(332)
            ax3 = fig.add_subplot(333)
            ax4 = fig.add_subplot(334)
            ax5 = fig.add_subplot(335)
            ax6 = fig.add_subplot(336)
            ax1.set(xlim=[0, n_epochs], title='train loss', xlabel='Epochs', ylabel='Loss')
            ax2.set(xlim=[0, n_epochs], title='joint loss', xlabel='Epochs', ylabel='Loss')
            ax3.set(xlim=[0, n_epochs], title='keypt loss', xlabel='Epochs', ylabel='Loss')
            ax4.set(xlim=[0, n_epochs], title='reg loss', xlabel='Epochs', ylabel='Loss')
            ax5.set(xlim=[0, n_epochs], title='shape loss', xlabel='Epochs', ylabel='Loss')
            ax6.set(xlim=[0, n_epochs], title='direc loss', xlabel='Epochs', ylabel='Loss')
        
            ax1.plot(train_loss.keys(), train_loss.values(), c = "r", linewidth=0.5)
            ax2.plot(j3d_loss.keys(), j3d_loss.values(), c = "r", linewidth=0.5)
            ax3.plot(j2d_loss.keys(), j2d_loss.values(), c = "r", linewidth=0.5)
            ax4.plot(reg_loss.keys(), reg_loss.values(), c = "r", linewidth=0.5)
            ax5.plot(shape_loss.keys(), shape_loss.values(), c = "r", linewidth=0.5)
            ax6.plot(direc_loss.keys(), direc_loss.values(), c = "r", linewidth=0.5)

            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax3.set_yscale('log')
            ax4.set_yscale('log')
            ax5.set_yscale('log')
            ax6.set_yscale('log')

            fig.savefig(os.path.join(self.save_path, 'logs', 'train loss.png'), facecolor='white')
            plt.close(fig)

            if self.cfg.fitting: 
                if self.cfg.lr_decay_gamma:
                    self.scheduler.step()

                if self.cfg.lr_reduce:
                    self.scheduler.step(valid_loss_dict['j2d_loss'])

                cur_lr = self.optimizer.param_groups[0]['lr']

                if cur_lr != prev_lr:
                    self.logger('====== Learning rate changed! %.2e -> %.2e ======' % (prev_lr, cur_lr))
                    prev_lr = cur_lr

            if self.cfg.validation:
                if valid_avg_meter.avg < best_loss:
                    best_model_dir = os.path.join(self.save_path, 'best_model')
                    if not os.path.exists(best_model_dir):
                        os.makedirs(best_model_dir)
                    self.cfg.best_model = os.path.join(best_model_dir, 'S%02d_%03d_net.pt' % (self.try_num, epoch_num))
                    self.save_model()
                    self.logger(f'Model saved! Try num : {self.try_num}, Epochs : {epoch_num}, Loss : {valid_avg_meter.avg}, Time : {datetime.now().replace(microsecond=0)}')
                    best_loss = valid_avg_meter.avg

                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(131)
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133)
                ax1.set(xlim=[0, n_epochs], title='train', xlabel='Epochs', ylabel='Loss')
                ax2.set(xlim=[0, n_epochs], title='valid', xlabel='Epochs', ylabel='Loss')
                ax3.set(xlim=[0, n_epochs], title='3D PCK', xlabel='Epochs', ylabel='3D PCK')
                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax1.plot(train_loss.keys(), train_loss.values(), c = "r", linewidth=0.5)
                ax2.plot(valid_loss.keys(), valid_loss.values(), c = "g", linewidth=0.5)
                ax3.plot(pck.keys(), pck.values(), c = "b", linewidth=0.5)
                fig.savefig(os.path.join(self.save_path, 'logs', 'train-valid loss.png'), facecolor='white')
                plt.close(fig)

        endtime = datetime.now().replace(microsecond=0)
        if self.cfg.validation:
            best_model_dir = os.path.join(self.save_path, 'best_model')
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            self.cfg.best_model = os.path.join(best_model_dir, 'S%02d_%03d_net.pt' % (self.try_num, epoch_num))
            self.save_model()
            self.logger('Best loss : %s\n' % best_loss)
            self.logger('Best model : %s\n' % self.cfg.best_model)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training time : %s\n' % (endtime - starttime))
        
if __name__ == "__main__":
    num_features = 2048
    num_param = 61

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible = False, width = 225, height = 225)
    config_num = 61
    configs = {
        'manual_seed' : 24756,
        'ckp_dir' : '/root/sensor-fusion-gesture/ckp/FreiHAND',
        # 'ckp_dir' : 'D:/sfGesture/ckp',
        'lr' : 0.0003958220000366005,
        'lr_decay_gamma' : 0.5,
        'lr_decay_step' : 50,
        'lr_reduce' : False,
        'expr_ID' : 'test1',
        'cuda_id' : 0,
        'dataset' : 'FreiHAND',
        'dataset_dir' : '/root/Dataset/FreiHAND_pub_v2',
        # 'dataset_dir' : 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB',
        'try_num' : 0,
        'optimizer' : "adam",
        'weight_decay' : 1e-5,
        'momentum' : 0,
        'use_multigpu' : True,
        'best_model' : None, 
        'num_workers' : 4, 
        'train_batch_size' : 64,
        'test_batch_size' : 16,
        'ckpt_term' : 100, 
        'n_epochs' : 100,
        'fitting' : True, 
        'depth_loss_weight': 0,
        'j2d_loss_weight' : 0.032723957226319454,
        'j3d_loss_weight' :0.09323343374985639,
        'reg_loss_weight' : 0.7386801944377812,
        'shape_loss_weight' : 3.370080749194146,
        'direc_loss_weight' : 0.0672720975418354,
        'scale_loss_weight' : 0,
        'normalize' : False,
        'num_iter' : 3,
        'pred_scale' : False,
        'num_fclayers' : [num_features+num_param, 
                        int(num_features),
                        int(num_features), 
                        num_param],
        'use_dropout' : [True,True,False],
        'drop_prob' : [0.8, 0.8, 0],
        'ac_func' : [True,True,False],
        'config_num': config_num,
        'iter' : False,
        'to_mano':None,
        'pretrained': True,
        'pretrained_net': False,
        # 'pretrained_net':"/root/Pretrained/S2HAND/texturehand_freihand.t7",
        'validation':True,
    } 

    config = configs
    cfg = Config(**config)
    path = os.path.join('/root/sensor-fusion-gesture/ckp/FreiHAND/results%d'%config_num)
    os.makedirs(path, exist_ok=True)
    cfg.write_cfg(write_path=os.path.join(path, 'config%d.yaml'%config_num))
    model = HMR(cfg)
    trainer = Trainer(cfg, model, vis)
    trainer.fit()

    vis.destroy_window()


    # combs = {0:[1e-1,1e2,1e2,1e-1], 1:[1e1,1e1,1e2,1e1], 2:[1e1,1e2,1e2,1e1], 3:[1e-1,1e3,1e3,1e-1], 4:[1e0,1e2,1e2,1e0]} 

    # j2d_loss_weight = [0.01, 0.1, 1, 10, 100] #5
    # j3d_loss_weight = [0.01, 0.1, 1, 10, 100] #5
    # reg_loss_weight = [0.01, 0.1, 1, 10, 100] #5
    # shape_loss_weight = [0.01, 0.1, 1, 10, 100] #5
    # direc_loss_weight = [0.01, 0.1, 1, 10, 100] #5
    # lr = [1e-3, 1e-4, 1e-5, 1e-6] #4
    # train_batch_size = [64, 128] #2

    # for i in range(5):
    #     config_num = 53 + i
    #     comb = combs[i]

    #     configs = {
    #         'manual_seed' : 24756,
    #         'ckp_dir' : '/root/sensor-fusion-gesture/ckp/FreiHAND',
    #         # 'ckp_dir' : 'D:/sfGesture/ckp',
    #         'lr' : 1e-5,
    #         'lr_decay_gamma' : 0.5,
    #         'lr_decay_step' : 50,
    #         'lr_reduce' : False,
    #         'expr_ID' : 'test1',
    #         'cuda_id' : 0,
    #         'dataset' : 'FreiHAND',
    #         'dataset_dir' : '/root/Dataset/FreiHAND_pub_v2',
    #         # 'dataset_dir' : 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB',
    #         'try_num' : 0,
    #         'optimizer' : "adam_amsgrad",
    #         'weight_decay' : 0,
    #         'momentum' : 0,
    #         'use_multigpu' : True,
    #         'best_model' : None, 
    #         'num_workers' : 4, 
    #         'train_batch_size' : 64,
    #         'test_batch_size' : 8,
    #         'ckpt_term' : 100, 
    #         'n_epochs' : 100,
    #         'fitting' : False, 
    #         'depth_loss_weight': 0,
    #         'j2d_loss_weight' : comb[0],
    #         'j3d_loss_weight' :comb[1],
    #         'reg_loss_weight' : comb[2],
    #         'shape_loss_weight' : 0,
    #         'direc_loss_weight' : comb[3],
    #         'scale_loss_weight' : 0,
    #         'normalize' : False,
    #         'num_iter' : 3,
    #         'pred_scale' : False,
    #         'num_fclayers' : [num_features+num_param, 
    #                         int(num_features), 
    #                         int(num_features),
    #                         num_param],
    #         'use_dropout' : [True,True,False],
    #         'drop_prob' : [0.5, 0.5, 0],
    #         'ac_func' : [True,True,False],
    #         'config_num': config_num,
    #         'iter' : False,
    #         'to_mano':None,
    #         'pretrained': True,
    #         'pretrained_net':False,
    #         'evaluation':False,
    #     } 

        

    #     config = configs
    #     cfg = Config(**config)
    #     path = os.path.join('/root/sensor-fusion-gesture/ckp/FreiHAND/results%d'%config_num)
    #     os.makedirs(path, exist_ok=True)
    #     cfg.write_cfg(write_path=os.path.join(path, 'config%d.yaml'%config_num))
    #     model = HMR(cfg)
    #     trainer = Trainer(cfg, model, vis)
    #     trainer.fit()

    # vis.destroy_window()



'''
        'j2d_loss_weight' : 1e-1,
        'j3d_loss_weight' :1e3,
        'reg_loss_weight' : 1e2,
        'shape_loss_weight' : 1e2,
        'direc_loss_weight' : 1e3,

        'j2d_loss_weight' : 0.01,
        'j3d_loss_weight' :1,
        'reg_loss_weight' : 1,
        'shape_loss_weight' : 0.01,
        'direc_loss_weight' : 0.01,
'''