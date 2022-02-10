"""
필요한가?
Trainting for get rotation of hand from mano hand model
Input : RGB(? x ?)
Output : rvec
"""
# ToDO : ADD joint loss!!!!
from ctypes.wintypes import tagMSG
import sys
from tracemalloc import start
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

from datasetloader.data_loader_MSRAHT import get_dataset
from net.hmr import HMR
from utils.train_utils import mklogger, AverageMeter, Mano2depth, Data_preprocess, set_vis, save_image, regularizer_loss
from utils.config_parser import Config
from net.hmr import HMR

class Trainer:
    def __init__(self, cfg, model):
        # Initialize randoms seeds
        torch.cuda.manual_seed_all(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        random.seed(cfg.manual_seed)

        starttime = datetime.now().replace(microsecond=0)

        #checkpoint dir
        log_path = os.makedirs(os.path.join(cfg.ckp_dir, 'logs'), exist_ok=True)
        logger = mklogger(log_path).info
        self.logger = logger
        os.makedirs(cfg.ckp_dir, exist_ok=True)
        summary_logdir = os.path.join(cfg.ckp_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('[%s] - Started training GrabNet, experiment code %s' % (cfg.expr_ID, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)
        logger('Base dataset_dir is %s' % cfg.dataset_dir)
        self.try_num = cfg.try_num


        self.model = model
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize optimizer
        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.5, 0.99)
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
                optimizer, patience=5, 
            )

        # loss
        self.depth_criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.joint_criterion = torch.nn.MSELoss(reduction='mean')

        # etc
        self.cfg = cfg
        self.start_epoch = 0
        self.vis = set_vis()

    def save_model(self):
        torch.save(self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) 
                    else self.model.state_dict(), self.cfg.best_model)

    def _get_model(self):
        pass

    def load_data(self, cfg):
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

        dat_name = cfg.dataset
        dat_dir = cfg.dataset_dir
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
            depth_image = sample['processed'].to(self.device) #[bs, 1, 224, 224]
            target_joint = sample['j3d'].to(self.device) #[bs,1,21,3]
            keypt, joint, vert, ang, faces, params = self.model(depth_image)
            # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
            m2d = Mano2depth(vert, faces)
            pred_depth = m2d.mesh2depth(self.vis) #[bs, 224, 224]
            # target_keypt = target_joint.squeeze()[:,:,:2] * params[:,0].contiguous().unsqueeze(1).unsqueeze(2) + params[:, 1:3].contiguous().unsqueeze(1)
            target_keypt = target_joint.squeeze()[:,:,:2]
            depth_loss = self.depth_criterion(pred_depth.to(self.device), depth_image.squeeze())
            j3d_loss = self.joint_criterion(joint.to(self.device), target_joint.squeeze())
            j2d_loss = self.joint_criterion(keypt.to(self.device), target_keypt)
            reg_loss = regularizer_loss(ang, params[:,6:16])

            # loss = depth_loss*1e6 + j2d_loss*1e1 +j3d_loss + reg_loss
            # loss = depth_loss*1e5 + j2d_loss + reg_loss*1e-1
            loss = depth_loss*1e4 + j2d_loss*1e-1 + reg_loss
            # loss = j3d_loss
            # loss = depth_loss
            train_loss_dict = {
                'depth_loss':depth_loss,
                'j3d_loss':j3d_loss,
                'j2d_loss':j2d_loss,
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

                print(
                    f'Step : {b_idx+1} / {len(self.train_dataloader)},' + \
                        f'Train loss : {avg_meter.avg:.5f},' + \
                            f'time: {term:.5f},', end = '\r'
                )

            self.optimizer.step()

        return avg_meter, train_loss_dict


    def eval(self):
        avg_meter = AverageMeter()
        self.model.eval()
        ckpt = 10
        t = time.time()

        for b_idx, (sample) in enumerate(self.test_dataloader):
            with torch.no_grad():
                depth_image = sample['processed'].to(self.device).float() #[bs, 1, 224, 224]
                coms = sample['com']
                target_joint = sample['j3d'].to(self.device) #[bs,1,21,3]
                name = sample['name']
                keypt, joint, vert, ang, faces, params = self.model(depth_image)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]

                if (b_idx+1) % ckpt == 0:
                    vis_path =  os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_pred_3d.png'%(self.start_epoch, b_idx+1))
                else:
                    vis_path = None

                m2d = Mano2depth(vert, faces)
                pred_depth = m2d.mesh2depth(self.vis, path = vis_path) #[bs, 224, 224]
                # target_keypt = target_joint.squeeze()[:,:,:2] * params[:,0].contiguous().unsqueeze(1).unsqueeze(2) + params[:, 1:3].contiguous().unsqueeze(1)
                target_keypt = target_joint.squeeze()[:,:,:2]
                depth_loss = self.depth_criterion(pred_depth.to(self.device), depth_image.squeeze())
                j3d_loss = self.joint_criterion(joint.to(self.device), target_joint.squeeze())
                j2d_loss = self.joint_criterion(keypt.to(self.device), target_keypt)
                reg_loss = regularizer_loss(ang, params[:,6:16])

                # loss = depth_loss*1e6 + j2d_loss*1e1 +j3d_loss + reg_loss*1e2
                # loss = depth_loss*1e5 + j2d_loss + reg_loss*1e-1
                loss = depth_loss*1e4 + j2d_loss*1e-1 + reg_loss
                # loss = depth_loss
                eval_loss_dict = {
                    'depth_loss':depth_loss,
                    'j3d_loss':j3d_loss,
                    'j2d_loss':j2d_loss,
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

                    print(
                        f'Step : {b_idx+1} / {len(self.test_dataloader)},' + \
                            f'Evaluation loss : {avg_meter.avg:.5f},' + \
                                f'time: {term:.5f},', end = '\r'
                    )  

                    pred = pred_depth[0]
                    target = depth_image[0]
                    pred_joint = joint[0].squeeze().cpu()
                    pred_vert = vert[0].squeeze().cpu()
                    vrot = params[0].squeeze().cpu()
                    pred_keypt = keypt[0].squeeze().cpu()
                    pred_faces = faces.detach().cpu()
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_joint.txt'%(self.start_epoch, b_idx+1)), pred_joint.numpy())
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_vert.txt'%(self.start_epoch, b_idx+1)), pred_vert.numpy())
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_faces.txt'%(self.start_epoch, b_idx+1)), pred_faces)
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_vrot.txt'%(self.start_epoch, b_idx+1)), vrot.numpy())
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_pred.txt'%(self.start_epoch, b_idx+1)), pred.squeeze().cpu().numpy())
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_target.txt'%(self.start_epoch, b_idx+1)), target.squeeze().cpu().numpy())
                    np.savetxt(os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_keypt.txt'%(self.start_epoch, b_idx+1)), pred_keypt.numpy())
                    save_image(pred, os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_pred_%s.png'%(self.start_epoch, b_idx+1, name[0])))
                    save_image(target, os.path.join(self.cfg.ckp_dir, 'results', 'E%d_%d_target_%s.png'%(self.start_epoch, b_idx+1, name[0])))

        return avg_meter, eval_loss_dict

    def fit(self, n_epochs = None):

        starttime = datetime.now().replace(microsecond=0)

        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr = np.inf
        best_loss = np.inf

        for epoch_num in range(1, n_epochs + 1):
            self.logger('===== starting Epoch # %03d' % epoch_num)

            train_avg_meter, train_loss_dict = self.train()
            print("[Epoch: %d/%d] Train loss : %.5f" % (epoch_num, n_epochs, train_avg_meter.avg))
            print("[Loss] depth_loss : %.5f, j3d_loss : %.5f, j2d_loss : %.5f, reg_loss : %.5f, total_loss : %.5f" % (
                train_loss_dict['depth_loss'],
                train_loss_dict['j3d_loss'],
                train_loss_dict['j2d_loss'],
                train_loss_dict['reg_loss'],
                train_loss_dict['total_loss']))
            
            eval_avg_meter, eval_loss_dict = self.eval()
            print("[Epoch: %d/%d] Evaluation loss : %.5f" % (epoch_num, n_epochs, eval_avg_meter.avg))
            print("[Loss] depth_loss : %.5f, j3d_loss : %.5f, j2d_loss : %.5f, reg_loss : %.5f, total_loss : %.5f" % (
                eval_loss_dict['depth_loss'],
                eval_loss_dict['j3d_loss'], 
                eval_loss_dict['j2d_loss'],
                eval_loss_dict['reg_loss'], 
                eval_loss_dict['total_loss']))

            if self.cfg.fitting: 
                if self.cfg.lr_decay_gamma:
                    self.scheduler.step()

                if self.cfg.lr_reduce:
                    self.scheduler.step(eval_loss_dict['total_loss'])

                cur_lr = self.optimizer.param_groups[0]['lr']

                if cur_lr != prev_lr:
                    self.logger('====== Learning rate changed! %.2e -> %.2e ======' % (prev_lr, cur_lr))
                    prev_lr = cur_lr

            if eval_avg_meter.avg < best_loss:
                best_model_dir = os.path.join(self.cfg.ckp_dir, 'best_model')
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                self.cfg.best_model = os.path.join(best_model_dir, 'S%02d_%03d_net.pt' % (self.try_num, epoch_num))
                self.save_model()
                self.logger(f'Model saved! Try num : {self.try_num}, Epochs : {epoch_num}, Loss : {eval_avg_meter.avg}, Time : {datetime.now().replace(microsecond=0)}')
                best_loss = eval_avg_meter.avg

        endtime = datetime.now().replace(microsecond=0)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training time : %s\n' % (endtime - starttime))
        self.logger('Best loss : %s\n' % best_loss)
        self.logger('Best model : %s\n' % self.cfg.best_model)


if __name__ == "__main__":
    config = {
        'manual_seed' : 23455,
        'ckp_dir' : '/root/sensor-fusion-gesture/ckp',
        # 'ckp_dir' : 'D:/sfGesture/ckp',
        'lr' : 1e-4,
        'lr_decay_gamma' : 0.1,
        'lr_decay_step' : 10,
        'lr_reduce' : False,
        'expr_ID' : 'test1',
        'cuda_id' : 0,
        'dataset' : 'MSRA_HT',
        'dataset_dir' : '/root/Dataset/cvpr14_MSRAHandTrackingDB',
        # 'dataset_dir' : 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB',
        'try_num' : 0,
        'optimizer' : 'adam',
        'weight_decay' : 0,
        'momentum' : 1.9,
        'use_multigpu' : True,
        'best_model' : None, 
        'num_workers' : 4, 
        'batch_size' : 40, 
        'ckpt_term' : 50, 
        'n_epochs' : 200,
        'fitting' : True,
    }

    cfg = Config(**config)
    model = HMR()
    trainer = Trainer(cfg, model)
    trainer.fit()