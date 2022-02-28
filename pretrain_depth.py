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

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from datasetloader.data_loader_MSRAHT import get_dataset
from net.hmr import HMR
from utils.train_utils import mklogger, AverageMeter, Mano2depth, Data_preprocess, set_vis, save_image, regularizer_loss, normalize
from utils.config_parser import Config
from net.hmr import HMR
from net.hmr_s2 import HMRS2, ResNet_Mano, BasicBlock, Bottleneck, DeconvBottleneck

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
        log_path = os.path.join(self.save_path, 'logs')
        os.makedirs(log_path, exist_ok=True)
        logger = mklogger(log_path).info
        self.logger = logger
        summary_logdir = os.path.join(self.save_path, 'summaries')
        self.swriter = SummaryWriter()
        logger('[%s] - Started training, experiment code %s' % (cfg.expr_ID, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)
        logger('Base dataset_dir is %s' % cfg.dataset_dir)
        self.try_num = cfg.try_num


        self.model = model
        if cfg.pretrained:
            self.load_model(model, cfg.pretrained)
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize optimizer
        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
            )
        elif cfg.optimizer == "rms":
            optimizer = torch.optim.RMSprop(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model_params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
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
        self.load_data(cfg, vis)

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
                optimizer, patience=1, 
            )

        # loss
        if cfg.SmmothL1loss_depth:
            self.depth_criterion = torch.nn.SmoothL1Loss(reduction='mean')
        if cfg.MSEloss_depth:
            self.depth_criterion = torch.nn.MSELoss(reduction='mean')

        self.criterion = torch.nn.MSELoss(reduction='mean')

        # etc
        self.cfg = cfg
        self.start_epoch = 0
        self.vis = vis

    def save_model(self):
        torch.save(self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) 
                    else self.model.state_dict(), self.cfg.best_model)

    def load_model(self, model, pretrained):
        model.load_state_dict(torch.load(pretrained))

    def load_data(self, cfg, vis=None):
        '''
        self.train_dataloader / self.val_dataloader ('depth','processed', 'cropped', 'trans')
        queries
        get_dataset(dat_name, set_name, base_path, queries, use_cache=True, train=False, split=None)
        ToDo : 1 dataset -> train / val split 

        return 0
        '''
        kwargs = {
            'num_workers' : cfg.num_workers,
            'batch_size' : cfg.batch_size,
            'shuffle' : True,
            'drop_last' : True,
        }

        dat_name = 'synthetic'
        dat_dir = cfg.dataset_dir #save path
        dataset = get_dataset(dat_name, dat_dir)

        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_dataloader = DataLoader(train_data, **kwargs)
        self.test_dataloader = DataLoader(test_data, **kwargs)
        print("Data size : ",len(self.train_dataloader), len(self.test_dataloader))


    def train(self):
        '''
        one epoch
        output(pse, bta, .... ) = model(input)
        vert, join = mano.forward(output..)
        mesh = o3d.geometry.TriangleMesh()
        ...
        mesh <predicted mesh> -> o3d.vis.capture_depth_... -> predicted depth(pred_depth)
        loss = SmoothL1loss(input, pred_depth)
        step(), backward() etc...
        '''
        avg_meter = AverageMeter()
        self.model.train()
        ckpt = 10
        t = time.time()
        self.start_epoch += 1

        for b_idx, (sample) in enumerate(self.train_dataloader):
            depth_image = sample['depth'].to(self.device) #[bs, 1, 224, 224]
            target_params = sample['params'].to(self.device) #[bs, 61]
            target_joints = sample['joints'].to(self.device)
            keypt, joint, vert, ang, faces, params = self.model(depth_image)
            # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
            m2d = Mano2depth(vert, faces)
            pred_depth = m2d.mesh2depth(self.vis) #[bs, 224, 224]
            # processed_param = self.model.module.get_theta_param(params) if isinstance(self.model, torch.nn.DataParallel) else self.model.get_theta_param(params)

            depth_loss = self.depth_criterion(pred_depth.to(self.device), depth_image.squeeze())
            reg_loss = torch.tensor(regularizer_loss(ang, theta = self.cfg.to_mano)).to(self.device)
            param_loss = self.criterion(params.to(self.device), target_params)
            joint_loss = self.criterion(joint.to(self.device), target_joints.squeeze())

            if self.cfg.depth_loss_weight != 0:
                depth_loss.requires_grad_(True)
            if self.cfg.reg_loss_weight != 0:
                reg_loss.requires_grad_(True)
            if self.cfg.params_loss_weight != 0:
                param_loss.requires_grad_(True)
            if self.cfg.joint_loss_weight != 0:
                joint_loss.requires_grad_(True)

            loss = depth_loss*self.cfg.depth_loss_weight + reg_loss*self.cfg.reg_loss_weight + param_loss*self.cfg.params_loss_weight + joint_loss*self.cfg.joint_loss_weight

            train_loss_dict = {
                'depth_loss':depth_loss,
                'param_loss' : param_loss,
                'joint_loss':joint_loss,
                'reg_loss':reg_loss,
                'total_loss':loss,
            }
            self.optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            avg_meter.update(loss.detach().item(), depth_image.shape[0])

            if (b_idx+1) % ckpt == 0:
                term = time.time() - t
                self.logger('Step : %s/%s' % (b_idx+1, len(self.train_dataloader)))
                self.logger('Train loss : %.5f' % avg_meter.avg)
                self.logger('time : %.5f' % term)
                self.logger("[Loss] depth_loss : %.5f, param_loss : %.5f, joint_loss : %.5f, reg_loss : %.5f, total_loss : %.5f" % (
                train_loss_dict['depth_loss'],
                train_loss_dict['param_loss'],
                train_loss_dict['joint_loss'],
                train_loss_dict['reg_loss'],
                train_loss_dict['total_loss']))

            self.optimizer.step()

        return avg_meter, train_loss_dict


    def eval(self):
        avg_meter = AverageMeter()
        self.model.eval()
        ckpt = 30
        t = time.time()

        for b_idx, (sample) in enumerate(self.test_dataloader):
            with torch.no_grad():
                depth_image = sample['depth'].to(self.device) #[bs, 1, 224, 224]
                target_params = sample['params'].to(self.device)
                target_joints = sample['joints'].to(self.device)
                name = sample['name']
                keypt, joint, vert, ang, faces, params = self.model(depth_image)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]

                if (b_idx+1) % ckpt == 0:
                    vis_path =  os.path.join(self.save_path, 'E%d_%d_pred_3d.png'%(self.start_epoch, b_idx+1))
                else:
                    vis_path = None

                m2d = Mano2depth(vert, faces)
                pred_depth = m2d.mesh2depth(self.vis, path = vis_path) #[bs, 224, 224]
                # processed_param = self.model.module.get_theta_param(params) if isinstance(self.model, torch.nn.DataParallel) else self.model.get_theta_param(params)

                depth_loss = self.depth_criterion(pred_depth.to(self.device), depth_image.squeeze())
                reg_loss = torch.tensor(regularizer_loss(ang, theta = self.cfg.to_mano)).to(self.device)
                param_loss = self.criterion(params.to(self.device), target_params)
                joint_loss = self.criterion(joint.to(self.device), target_joints.squeeze())

                if self.cfg.depth_loss_weight != 0:
                    depth_loss.requires_grad_(True)
                if self.cfg.reg_loss_weight != 0:
                    reg_loss.requires_grad_(True)
                if self.cfg.params_loss_weight != 0:
                    param_loss.requires_grad_(True)
                if self.cfg.joint_loss_weight != 0:
                    joint_loss.requires_grad_(True)

                loss = depth_loss*self.cfg.depth_loss_weight + reg_loss*self.cfg.reg_loss_weight + param_loss*self.cfg.params_loss_weight + joint_loss*self.cfg.joint_loss_weight

                eval_loss_dict = {
                    'depth_loss':depth_loss,
                    'param_loss':param_loss,
                    'joint_loss':joint_loss,
                    'reg_loss':reg_loss,
                    'total_loss':loss,
                }

                loss.requires_grad_(True)
                avg_meter.update(loss.detach().item(), depth_image.shape[0])

                if (b_idx+1) % ckpt == 0:
                    term = time.time() - t
                    self.logger('Step : %s/%s' % (b_idx+1, len(self.test_dataloader)))
                    self.logger('Evaluation loss : %.5f' % avg_meter.avg)
                    self.logger('time : %.5f' % term)
                    self.logger("[Loss] depth_loss : %.5f, param_loss : %.5f, joint_loss : %.5f, reg_loss : %.5f, total_loss : %.5f" % (
                    eval_loss_dict['depth_loss'],
                    eval_loss_dict['param_loss'], 
                    eval_loss_dict['joint_loss'],
                    eval_loss_dict['reg_loss'], 
                    eval_loss_dict['total_loss']))

                    pred = pred_depth[0]
                    target = depth_image[0]
                    param = params[0].squeeze().cpu()
                    target_param = target_params[0].squeeze().cpu()

                    np.savetxt(os.path.join(self.save_path, 'E%d_%d_param.txt'%(self.start_epoch, b_idx+1)), param.numpy())
                    np.savetxt(os.path.join(self.save_path, 'E%d_%d_targetparam.txt'%(self.start_epoch, b_idx+1)), target_param.numpy())
                    np.savetxt(os.path.join(self.save_path, 'E%d_%d_pred.txt'%(self.start_epoch, b_idx+1)), pred.squeeze().cpu().numpy())
                    np.savetxt(os.path.join(self.save_path, 'E%d_%d_target.txt'%(self.start_epoch, b_idx+1)), target.squeeze().cpu().numpy())
                    save_image(pred, os.path.join(self.save_path, 'E%d_%d_pred_%s.png'%(self.start_epoch, b_idx+1, name[0])))
                    save_image(target, os.path.join(self.save_path, 'E%d_%d_target_%s.png'%(self.start_epoch, b_idx+1, name[0])))

        return avg_meter, eval_loss_dict

    def fit(self, n_epochs = None):

        starttime = datetime.now().replace(microsecond=0)
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        ax1.set(xlim=[0, n_epochs], title='Train Loss', xlabel='Epochs', ylabel='Loss')
        ax2.set(xlim=[0, n_epochs], title='Evaluation Loss', xlabel='Epochs', ylabel='Loss')

        self.logger('Started training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr = np.inf
        best_loss = np.inf

        for epoch_num in range(1, n_epochs + 1):
            self.logger('===== starting Epoch # %03d' % epoch_num)

            train_avg_meter, train_loss_dict = self.train()
            self.logger("[Epoch: %d/%d] Train loss : %.5f" % (epoch_num, n_epochs, train_avg_meter.avg))
            self.logger("[Loss] depth_loss : %.5f, param_loss : %.5f, joint_loss : %.5f, reg_loss : %.5f, total_loss : %.5f" % (
                train_loss_dict['depth_loss'],
                train_loss_dict['param_loss'],
                train_loss_dict['joint_loss'],
                train_loss_dict['reg_loss'],
                train_loss_dict['total_loss']))
            ax1.scatter(epoch_num, train_avg_meter.avg)
            
            eval_avg_meter, eval_loss_dict = self.eval()
            self.logger("[Epoch: %d/%d] Evaluation loss : %.5f" % (epoch_num, n_epochs, eval_avg_meter.avg))
            self.logger("[Loss] depth_loss : %.5f, param_loss : %.5f, joint_loss : %.5f, reg_loss : %.5f, total_loss : %.5f" % (
                eval_loss_dict['depth_loss'],
                eval_loss_dict['param_loss'], 
                eval_loss_dict['joint_loss'],
                eval_loss_dict['reg_loss'], 
                eval_loss_dict['total_loss']))
            ax2.scatter(epoch_num, eval_avg_meter.avg)

            if self.cfg.fitting: 
                if self.cfg.lr_decay_gamma:
                    self.scheduler.step()

                if self.cfg.lr_reduce:
                    self.scheduler.step(eval_loss_dict['j3d_loss'])

                cur_lr = self.optimizer.param_groups[0]['lr']

                if cur_lr != prev_lr:
                    self.logger('====== Learning rate changed! %.2e -> %.2e ======' % (prev_lr, cur_lr))
                    prev_lr = cur_lr

            if eval_avg_meter.avg < best_loss:
                best_model_dir = os.path.join(self.save_path, 'best_model')
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                self.cfg.best_model = os.path.join(best_model_dir, 'S%02d_%03d_net.pt' % (self.try_num, epoch_num))
                self.save_model()
                self.logger(f'Model saved! Try num : {self.try_num}, Epochs : {epoch_num}, Loss : {eval_avg_meter.avg}, Time : {datetime.now().replace(microsecond=0)}')
                best_loss = eval_avg_meter.avg

        endtime = datetime.now().replace(microsecond=0)
        fig.savefig(os.path.join(self.save_path, 'loss.png'), facecolor='white')
        plt.close(fig)
        best_model_dir = os.path.join(self.save_path, 'best_model')
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        self.cfg.best_model = os.path.join(best_model_dir, 'S%02d_%03d_net.pt' % (self.try_num, epoch_num))
        self.save_model()
        self.logger(f'Model saved! Try num : {self.try_num}, Epochs : {epoch_num}, Loss : {eval_avg_meter.avg}, Time : {datetime.now().replace(microsecond=0)}')
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training time : %s\n' % (endtime - starttime))
        self.logger('Best loss : %s\n' % best_loss)
        self.logger('Best model : %s\n' % self.cfg.best_model)


if __name__ == "__main__":
    num_features = 2048
    num_param = 39
    
    configs = {
        'manual_seed' : 21846,
        'ckp_dir' : '/root/sensor-fusion-gesture/pretrain_ckp',
        # 'ckp_dir' : 'D:/sfGesture/ckp',
        'lr' : 0.1,
        'lr_decay_gamma' : 0.1,
        'lr_decay_step' : 5,
        'lr_reduce' : False,
        'expr_ID' : 'test1',
        'cuda_id' : 0,
        'dataset' : 'synthetic',
        'dataset_dir' : '/root/Dataset/synthetic/003', # database save dir
        # 'dataset_dir' : 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB',
        'try_num' : 0,
        'optimizer' : "adam",
        'weight_decay' : 0.1,
        'momentum' : 0.9,
        'use_multigpu' : True,
        'best_model' : None, 
        'num_workers' : 4, 
        'batch_size' : 32,
        'ckpt_term' : 100, 
        'n_epochs' : 100,
        'fitting' : True,
        'depth_loss_weight': 0,
        'j2d_loss_weight' : 0,
        'j3d_loss_weight' :0,
        'reg_loss_weight' : 0,
        'params_loss_weight' : 1,
        'joint_loss_weight' : 0, 
        'normalize' : False,
        'SmmothL1loss_depth' : True,
        'MSEloss_depth' : False,
        'num_iter' : 3,
        'pred_scale' : False,
        'num_fclayers' : [num_features+num_param, 
                         int(num_features/4), 
                         int(num_features/4),
                         num_param],
        'use_dropout' : [True,True,False],
        'drop_prob' : [0.5, 0.5, 0],
        'ac_func' : [True,True,False],
        # 'model_pretrained': '/root/sensor-fusion-gesture/pretrain_ckp/results0/best_model/S00_019_net.pt',
        'pretrained' : False,
        'iter' : False,
        'to_mano' : None,
        'config_num' : 4,
    } 

    
    vis = set_vis()

    cfg = Config(**configs)
    cfg.write_cfg(write_path=os.path.join('./pretrain_ckp', 'config4.yaml'))
    model = HMR(cfg)
    trainer = Trainer(cfg, model, vis)
    trainer.fit()

    vis.destroy_window()