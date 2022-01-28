"""
필요한가?
Trainting for get rotation of hand from mano hand model
Input : RGB(? x ?)
Output : rvec
"""
# ToDO : ADD joint loss!!!!
import sys
from tracemalloc import start
sys.path.append('.')
sys.path.append('..')
# sys.path.append("C:\\Users\\UVRLab\\Desktop\\sfGesture")
sys.path.append('/root/sensor-fusion-gesture')
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
from utils.train_utils import mklogger, AverageMeter, Mano2depth, Data_preprocess, set_vis, save_image
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
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if use_cuda else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = torch.cuda.device_count() if cfg.use_multigpu else 1
        if use_cuda:
            logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        self.model.to(self.device)
        if cfg.use_multigpu:
            self.model = torch.nn.DataParallel(self.model)
            logger("Training on Multiple GPU's")

        self.load_data(cfg)

        # load learning rate
        for group in optimizer.param_groups:
            group['lr'] = cfg.lr
            group['initial_lr'] = cfg.lr

        if cfg.lr_decay_gamma:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, cfg.lr_decay_step, gamma = cfg.lr_decay_gamma
                )

        # loss
        self.loss = torch.nn.SmoothL1Loss()

        # etc
        self.cfg = cfg
        self.vis = set_vis()
        self.start_epoch = 0

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
        # ckpt = len(self.test_dataloader) / self.cfg.ckpt_term
        ckpt = 10
        loss_train_set = []
        t = time.time()
        self.start_epoch += 1

        for b_idx, (sample) in enumerate(self.train_dataloader):
            depth_image = sample['processed'].to(self.device) #[bs, 1, 224, 224]
            coms = sample['com']
            self.optimizer.zero_grad()
            keypt, joint, vert, ang, faces, params = self.model(depth_image)
            # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
            m2d = Mano2depth(vert, faces)
            pred_depth = m2d.mesh2depth(self.vis, coms) #[bs, 224, 224]
            loss = self.loss(depth_image.squeeze(), pred_depth.to(self.device))
            loss_train_set.append(loss.item())
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

                print(f'Train loss2 : {np.mean(loss_train_set)}')

            self.optimizer.step()

        return loss_train_set, avg_meter


    def eval(self):
        avg_meter = AverageMeter()
        self.model.eval()
        # ckpt = len(self.test_dataloader) / self.cfg.ckpt_term
        ckpt = 10
        loss_eval_set = []
        t = time.time()

        for b_idx, (sample) in enumerate(self.test_dataloader):
            with torch.no_grad():
                depth_image = sample['processed'].to(self.device).float() #[bs, 1, 224, 224]
                coms = sample['com']
                self.optimizer.zero_grad()
                keypt, joint, vert, ang, faces, params = self.model(depth_image)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
                m2d = Mano2depth(vert, faces)

                pred_depth = m2d.mesh2depth(self.vis, coms) #[bs, 224, 224]
                loss = self.loss(depth_image.squeeze(), pred_depth.to(self.device))
                loss.requires_grad_(True)
                loss_eval_set.append(loss.item())
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

                    print(f'Evaluation loss2 : {np.mean(loss_eval_set)}')   

                    pred = pred_depth[0]
                    target = depth_image[0]
                    save_image(pred, os.path.join(self.cfg.ckp_dir, 'results', 'E%3d_%3d_pred.png'%(self.start_epoch, b_idx+1)))
                    save_image(target, os.path.join(self.cfg.ckp_dir, 'results', 'E%3d_%3d_target.png'%(self.start_epoch, b_idx+1)))

        return loss_eval_set, avg_meter

    def fit(self, n_epochs = None):

        starttime = datetime.now().replace(microsecond=0)

        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr = np.inf
        best_loss = np.inf
        loss_train_sets = []
        loss_eval_sets = []

        for epoch_num in range(1, n_epochs + 1):
            self.logger('===== starting Epoch # %03d' % epoch_num)

            loss_train_sets.append([])
            loss_eval_sets.append([])

            loss_train_set, train_avg_meter = self.train()
            loss_train_sets[-1].append(loss_train_set)

            if train_avg_meter.avg != np.mean(loss_train_sets[-1]):
                print("Something wrong") #나중에 없앰

            print("[Epoch: %d/%d] Train loss : %.5f" % (epoch_num, n_epochs, train_avg_meter.avg))
            
            loss_eval_set, eval_avg_meter = self.eval()
            loss_eval_sets[-1].append(loss_eval_set)
            print("[Epoch: %d/%d] Evaluation loss : %.5f" % (epoch_num, n_epochs, eval_avg_meter.avg))

            if self.cfg.fitting: 
                if self.cfg.lr_decay_gamma:
                    self.scheduler.step()
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
                self.best_loss = eval_avg_meter.avg

        endtime = datetime.now().replace(microsecond=0)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training time : %s\n' % (endtime - starttime))
        self.logger('Best loss : %s\n' % best_loss)
        self.logger('Best model : %s\n' % self.cfg.best_model)


if __name__ == "__main__":
    config = {
        'manual_seed' : 24587,
        # 'ckp_dir' : '/root/sensor-fusion-gesture/ckp',
        'ckp_dir' : './ckp',
        'lr' : 0.1,
        'lr_decay_gamma' : 1e-6,
        'lr_decay_step' : 10,
        'expr_ID' : 'test1',
        'cuda_id' : 0,
        'dataset' : 'MSRA_HT',
        # 'dataset_dir' : '/root/Dataset/cvpr14_MSRAHandTrackingDB',
        'dataset_dir' : 'D:/datasets/cvpr14_MSRAHandTrackingDB/cvpr14_MSRAHandTrackingDB',
        'try_num' : 0,
        'optimizer' : 'sgd',
        'weight_decay' : 0,
        'momentum' : 1.9,
        'use_multigpu' : False,
        'best_model' : None, 
        'num_workers' : 2, 
        'batch_size' : 20, 
        'ckpt_term' : 100, 
        'n_epochs' : 50,
        'fitting' : True,
    }

    cfg = Config(**config)
    model = HMR()
    trainer = Trainer(cfg, model)
    trainer.fit()