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
from net.hmr import HMR_tuning
from utils.train_utils import mklogger, AverageMeter, Mano2depth, Data_preprocess, ResultGif, proj_func, set_vis, save_image, regularizer_loss, shape_loss, direction_loss, normalize, proj_func
from utils.config_parser import Config

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

class Trainer:
    def __init__(self, config, model):
        # Initialize randoms seeds
        torch.cuda.manual_seed_all(config["manual_seed"])
        torch.manual_seed(config["manual_seed"])
        np.random.seed(config["manual_seed"])
        random.seed(config["manual_seed"])

        starttime = datetime.now().replace(microsecond=0)

        #checkpoint dir
        os.makedirs(config["ckp_dir"], exist_ok=True)
        self.save_path = os.path.join(config["ckp_dir"],'results%d'%config["config_num"])
        os.makedirs(self.save_path, exist_ok=True)
        #===
        os.makedirs(os.path.join(self.save_path, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'target'), exist_ok=True)
        #===
        log_path = os.path.join(self.save_path, 'logs')
        os.makedirs(log_path, exist_ok=True)
        logger = mklogger(log_path).info
        self.logger = logger
        summary_logdir = os.path.join(self.save_path, 'summaries')
        # self.swriter = SummaryWriter()
        logger('[%s] - Started training, experiment code %s' % (config["expr_ID"], starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)
        logger('Base dataset_dir is %s' % config["dataset_dir"])
        self.try_num = config["try_num"]


        self.model = model
        if config["pretrained_net"]:
            self.load_model(model, config["pretrained"])
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize optimizer
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                model_params, lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.999)
            )
        elif config["optimizer"] == "adam_amsgrad":
            optimizer = torch.optim.Adam(
                model_params, lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.999), amsgrad=True
            )
        elif config["optimizer"] == "rms":
            optimizer = torch.optim.RMSprop(
                model_params, lr=config["lr"], weight_decay=config["weight_decay"]
            )
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                model_params,
                lr=config["lr"],
                # momentum=config["momentum"],
                # weight_decay=config["weight_decay"],
            )
        self.optimizer = optimizer

        # cuda 
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % config["cuda_id"] if use_cuda else "cpu")

        gpu_brand = torch.cuda.get_device_name(config["cuda_id"]) if use_cuda else None
        gpu_count = torch.cuda.device_count() if config["use_multigpu"] else 1
        if use_cuda:
            logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        
        if config["use_multigpu"]:
            self.model = nn.DataParallel(self.model)
            logger("Training on Multiple GPU's")

        self.model = self.model.to(self.device)
        self.load_data(config)

        # load learning rate
        for group in optimizer.param_groups:
            group['lr'] = config["lr"]
            group['initial_lr'] = config["lr"]

        if config["lr_decay_gamma"]:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, config["lr_decay_step"], gamma = config["lr_decay_gamma"]
                )

        if config["lr_reduce"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, 
            )

        # loss
        self.smoothl1_criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')

        # etc
        self.config = config        
        self.start_epoch = 0
        self.max_norm = 5

    def save_model(self):
        torch.save(self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) 
                    else self.model.state_dict(), self.config["best_model"])

    def load_model(self, model, pretrained):
        model.load_state_dict(torch.load(pretrained))

    def load_data(self, config):
        '''
        self.train_dataloader / self.val_dataloader ('depth','processed', 'cropped', 'trans')
        queries
        get_dataset(dat_name, set_name, base_path, queries, use_cache=True, train=False, split=None)
        ToDo : 1 dataset -> train / val split 

        return 0
        '''
        kwargs_train = {
            'num_workers' : config["num_workers"],
            'batch_size' : config["train_batch_size"],
            'shuffle' : True,
            'drop_last' : True,
        }

        kwargs_test = {
            'num_workers' : config["num_workers"],
            'batch_size' : config["test_batch_size"],
            'shuffle' : True,
            'drop_last' : True,
        }

        dat_name = config["dataset"]
        dat_dir = config["dataset_dir"]
        # train_queries = ['image', 'open_j2d', 'Ks', 'j3d', 'verts', 'mano']
        train_queries = ['trans_img', 'trans_open_j2d', 'trans_Ks', 'trans_j3d', 'trans_verts', 'mano']
        evaluation_queries = ['image', 'Ks', 'open_j2d']
        train_dataset = get_dataset(dat_name, dat_dir, queries=train_queries, set_name='training')
        # test_dataset = get_dataset(dat_name, dat_dir, queries=evaluation_queries, set_name='evaluation')
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [25600, len(train_dataset) - 25600])
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
            target_open_j2d = sample['trans_open_j2d'].to(self.device) #[bs, 21, 2]
            target_mano = sample['mano'].to(self.device) #[bs, 1, 61]
            Ks = sample['trans_Ks'].to(self.device) #[bs, 3, 3]

            keypt, joint, vert, ang, pose, faces, params = self.model(target_img, evaluation=False)
            # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [bs, 15, 3], [1538,3], [bs, 61]
            beta = params[:,6:16].contiguous()

            target_joint_norm = normalize(target_joint)
            joint_norm = normalize(joint)

            target_keypt = proj_func(target_joint, Ks)
            target_keypt_norm = normalize(target_keypt, mode='keypt')
            keypt_norm = normalize(keypt, mode='keypt')

            j3d_loss = self.mse_criterion(joint_norm.to(self.device), target_joint_norm.squeeze())
            j2d_loss = self.smoothl1_criterion(keypt_norm.to(self.device), target_keypt_norm.squeeze())
            # scale_loss = self.smoothl1_criterion(params[:, 0].to(self.device), target_mano.squeeze()[:,-6])
            # open_j2d_loss = self.joint_criterion(keypt.to(self.device), target_open_j2d.squeeze())
            reg_loss = regularizer_loss(pose)
            beta_loss = self.mse_criterion(beta, torch.zeros_like(beta).to(self.device))
            direc_loss = direction_loss(target_joint_norm, joint)

            loss = j3d_loss*self.config["j3d_loss_weight"] + \
                j2d_loss*self.config["j2d_loss_weight"] + \
                reg_loss*self.config["reg_loss_weight"] + \
                beta_loss*self.config["shape_loss_weight"] + \
                direc_loss*self.config["direc_loss_weight"]
                # scale_loss*config["scale_loss_weight

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
        ckpt = 10
        t = time.time()
        pck_threshold = 1.0

        for b_idx, (sample) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                target_img = sample['trans_img'].to(self.device) #[bs, 3, 224, 224]
                target_joint = sample['trans_j3d'].to(self.device) #[bs,21,3]
                target_open_j2d = sample['trans_open_j2d'].to(self.device) #[bs, 21, 2]
                target_mano = sample['mano'].to(self.device) #[bs, 1, 61]
                Ks = sample['trans_Ks'].to(self.device) #[bs, 3, 3]

                keypt, joint, vert, ang, pose, faces, params = self.model(target_img)
                # [bs, 21, 2], [bs, 21, 3], [bs, 778, 3], [bs, 23], [bs, 15, 3], [1538,3], [bs, 61]
                beta = params[:,6:16].contiguous()
                target_keypt = proj_func(target_joint, Ks)

                target_joint_norm = normalize(target_joint)
                joint_norm = normalize(joint)

                target_keypt_norm = normalize(target_keypt, mode='keypt')
                keypt_norm = normalize(keypt, mode='keypt')

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

                loss = j3d_loss*self.config["j3d_loss_weight"]+ \
                    j2d_loss*self.config["j2d_loss_weight"] + \
                    reg_loss*self.config["reg_loss_weight"] + \
                    beta_loss*self.config["shape_loss_weight"] + \
                    direc_loss*self.config["direc_loss_weight"]
                    # scale_loss*config["scale_loss_weight

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
            n_epochs = self.config["n_epochs"]

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

            if self.config["validation"]:
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
            
            tune.report(loss=valid_avg_meter.avg, accuracy=pck_meter.avg)

            if self.config["fitting"]: 
                if self.config["lr_decay_gamma"]:
                    self.scheduler.step()

                if self.config["lr_reduce"]:
                    self.scheduler.step(valid_loss_dict['j2d_loss'])

                cur_lr = self.optimizer.param_groups[0]['lr']

                if cur_lr != prev_lr:
                    self.logger('====== Learning rate changed! %.2e -> %.2e ======' % (prev_lr, cur_lr))
                    prev_lr = cur_lr

        self.logger('Finished Training')


def main(config, vis=None):
    model = HMR_tuning(config)
    trainer = Trainer(config, model)
    trainer.fit()

if __name__ == "__main__":
    configs = {
        'manual_seed' : 24756,
        'ckp_dir' : '/root/sensor-fusion-gesture/ckp/FreiHAND',
        # 'ckp_dir' : 'D:/sfGesture/ckp',
        'lr' : tune.loguniform(1e-6, 1e-3),
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
        'weight_decay' : tune.choice([0.1, 1e-3, 1e-5]),
        'momentum' : 0,
        'use_multigpu' : True,
        'best_model' : None, 
        'num_workers' : 4, 
        'train_batch_size' : 64,
        'test_batch_size' : 16,
        'ckpt_term' : 100, 
        'n_epochs' : 10,
        'fitting' : True, 
        'depth_loss_weight': 0,
        'j2d_loss_weight' : tune.loguniform(1e-2, 1e4),
        'j3d_loss_weight' : tune.loguniform(1e-2, 1e4),
        'reg_loss_weight' : tune.loguniform(1e-2, 1e4),
        'shape_loss_weight' : tune.loguniform(1e-2, 1e4),
        'direc_loss_weight' : tune.loguniform(1e-2, 1e4),
        'scale_loss_weight' : 0,
        'normalize' : False,
        'num_iter' : 3,
        'pred_scale' : False,
        'num_fclayers' : [2048+61, 
                        int(2048),
                        int(2048), 
                        61],
        'use_dropout' : [True,True,False],
        'drop_prob' : [0.5, 0.5, 0],
        'ac_func' : [True,True,False],
        'config_num': 60,
        'iter' : False,
        'to_mano':None,
        'pretrained': True,
        'pretrained_net':False,
        'validation':True,
    } 

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        main,
        resources_per_trial={"cpu": 2, "gpu": 2},
        config=configs,
        num_samples=30,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    f = open('/root/sensor-fusion-gesture/ckp/FreiHAND/results60/logs/best_trial.txt', 'w')
    print("Best trial config: {}".format(best_trial.config), file=f)
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]), file=f)
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]), file=f)
    f.close()
