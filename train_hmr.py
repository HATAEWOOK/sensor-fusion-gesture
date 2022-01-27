"""
필요한가?
Trainting for get rotation of hand from mano hand model
Input : RGB(? x ?)
Output : rvec
"""

import sys
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
from datetime import datetime
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasetloader.data_loader_MSRAHT import get_dataset
from net.hmr import HMR
from utils.train_utils import mklogger, AverageMeter, Mano2depth, Data_preprocess

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


        self.model = model
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize optimizer
        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model_params, lr=cfg.lr, weight_decay=cfg.weight_decay
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
            gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
            gpu_count = torch.cuda.device_count() if cfg.use_multigpu else 1
            logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        self.device = torch.device("cuda%d" % cfg.cuda_id if use_cuda else "cpu")
        self.model.to(self.device)
        if cfg.use_multigpu:
            self.model = torch.nn.DataParallel(self.model)
            logger("Training on Multiple GPU's")

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

    def save_model(self):
        pass

    def _get_model(self):
        pass

    def load_dat(self, cfg):
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
        ckpt = len(self.train_dataloader) / self.cfg.ckpt_term
        loss_train_set = []

        for b_idx, (sample) in enumerate(self.train_dataloader):
            #ToDo ckpt에 따라 self.logger(train info, loss info)
            depth_image = sample['processed'].to(self.device).float() #[bs, 1, 224, 224]
            self.optimizer.zero_grad()
            keypt, joint, vert, ang, faces, params = self.model(depth_image)
            # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
            m2d = Mano2depth(vert, faces)

            pred_depth = m2d.mesh2depth() #[bs, 224, 224]

            loss = self.loss(depth_image.squeeze(), pred_depth)
            loss_train_set[-1].append(loss.item())
            loss.backward()
            avg_meter.update(loss.detatch().item(), depth_image.shape[0])

            self.optimizer.step()
            if self.cfg.lr_decay_gamma:
                self.scheduler.step()

        return loss_train_set, avg_meter


    def eval(self):
        avg_meter = AverageMeter()
        self.model.eval()
        ckpt = len(self.test_dataloader) / self.cfg.ckpt_term
        loss_eval_set = []

        for b_idx, (sample) in enumerate(self.test_dataloader):
            #ToDo ckpt에 따라 self.logger(train info, loss info)
            with torch.no_grad():
                depth_image = sample['processed'].to(self.device).float() #[bs, 1, 224, 224]
                self.optimizer.zero_grad()
                keypt, joint, vert, ang, faces, params = self.model(depth_image)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [1538,3], [bs, 39]
                m2d = Mano2depth(vert, faces)

                pred_depth = m2d.mesh2depth() #[bs, 224, 224]

                loss = self.loss(depth_image.squeeze(), pred_depth)
                loss_eval_set[-1].append(loss.item())
                avg_meter.update(loss.detatch().item(), depth_image.shape[0])

        return loss_eval_set, avg_meter

    def fit(self):
        pass



