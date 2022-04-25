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
from utils.train_utils import mklogger, AverageMeter, Mano2depth, proj_func, save_image, regularizer_loss, direction_loss, normalize, proj_func, orthographic_proj_withz, compute_uv_from_integral, normalize_image
from utils.config_parser import Config
from net.hmr_s2 import RGB2HM

class Trainer:
    def __init__(self, cfg, model):
        # Initialize randoms seeds
        torch.cuda.manual_seed_all(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        random.seed(cfg.manual_seed)
        torch.backends.cudnn.benchmark = True

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
            Ks = sample['trans_Ks'].to(self.device) #[bs, 3, 3]

            input = normalize_image(target_img)
            if input.shape[3] != 256:
                pad = nn.ZeroPad2d(padding=(0,32,0,32))
                input = pad(input)
            hm_list, encoding = self.model(input)
            hm_keypt_list = []
            for hm in hm_list:
                hm_keypt = compute_uv_from_integral(hm, input.shape[2:4])
                hm_keypt_list.append(hm_keypt)

            hm_keypt = hm_keypt_list[1]
            hm_2d_keypt = hm_keypt[:,:,:2]
            target_keypt = proj_func(target_joint, Ks)
            #ToDo:do not normalize
            target_keypt_norm = normalize(target_keypt, mode='keypt') #[bs,3]
            hm_2d_keypt_norm = normalize(hm_2d_keypt, mode='keypt')

            # j2d_loss = self.smoothl1_criterion(hm_2d_keypt_norm.to(self.device), target_keypt_norm.squeeze())
            j2d_loss = self.smoothl1_criterion(hm_2d_keypt.to(self.device), target_keypt.squeeze())

            loss = j2d_loss*self.cfg.j2d_loss_weight 

            train_loss_dict = {
                'j2d_loss':j2d_loss,
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
                self.logger("[Loss] j2d_loss : %.5f, total_loss : %.5f" % (
                train_loss_dict['j2d_loss'],
                train_loss_dict['total_loss']))


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
                Ks = sample['trans_Ks'].to(self.device) #[bs, 3, 3]

                input = normalize_image(target_img)
                if input.shape[3] != 256:
                    pad = nn.ZeroPad2d(padding=(0,32,0,32))
                    input = pad(input)
                hm_list, encoding = self.model(input)
                hm_keypt_list = []
                for hm in hm_list:
                    hm_keypt = compute_uv_from_integral(hm, input.shape[2:4])
                    hm_keypt_list.append(hm_keypt)

                hm_keypt = hm_keypt_list[1]
                hm_2d_keypt = hm_keypt[:,:,:2]

                target_keypt = proj_func(target_joint, Ks)

                target_keypt_norm = normalize(target_keypt, mode='keypt')
                hm_2d_keypt_norm = normalize(hm_2d_keypt, mode='keypt')

                if (b_idx+1) % ckpt == 0 and self.start_epoch % 10 == 0:
                # if (b_idx+1) % ckpt == 0:
                    img = target_img[0].cpu().permute(1,2,0).numpy()
                    heatmap = hm_2d_keypt[0].detach().cpu().numpy()
                    keypt = target_keypt[0].detach().cpu().numpy()
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    ax1.scatter(heatmap[:,0], heatmap[:,1], marker='o', c = 'r')
                    ax2.scatter(keypt[:,0], keypt[:,1], marker='o', c = 'b')
                    ax1.imshow(img, cmap='gray')
                    ax2.imshow(img, cmap='gray')
                    ax1.axis('off')
                    ax2.axis('off')
                    fig.savefig(os.path.join(self.save_path, 'img', 'E%d_%d_heatmap.png'%(self.start_epoch, b_idx+1)))
                    plt.close(fig)

                joint_distance = torch.norm(target_keypt - hm_2d_keypt, dim = 2) #[bs, 21] distance of each joints
                joint_index = torch.where(joint_distance > pck_threshold, torch.zeros_like(joint_distance), torch.ones_like(joint_distance))
                pck = torch.mean(joint_index)
                pck_meter.update(pck.item())

                # j2d_loss = self.smoothl1_criterion(hm_2d_keypt_norm.to(self.device), target_keypt_norm.squeeze())
                j2d_loss = self.smoothl1_criterion(hm_2d_keypt.to(self.device), target_keypt.squeeze())

                loss = j2d_loss*self.cfg.j2d_loss_weight

                valid_loss_dict = {
                    'j2d_loss':j2d_loss,
                    'total_loss':loss,
                }
                
                avg_meter.update(loss.item())

                if (b_idx+1) % ckpt == 0:
                    term = time.time() - t
                    self.logger('Step : %s/%s' % (b_idx+1, len(self.train_dataloader)))
                    self.logger('Train loss : %.5f' % avg_meter.avg)
                    self.logger('time : %.5f' % term)
                    self.logger("[Loss] j2d_loss : %.5f, total_loss : %.5f" % (
                    valid_loss_dict['j2d_loss'],
                    valid_loss_dict['total_loss']))
                    self.logger("[2D PCK] 2D PCK : %.3f"%pck_meter.avg)

        return avg_meter, valid_loss_dict, pck_meter

    def fit(self, n_epochs = None):

            starttime = datetime.now().replace(microsecond=0)
            train_loss = {}
            j2d_loss = {}
            valid_loss = {}
            pck = {}

            if n_epochs is None:
                n_epochs = self.cfg.n_epochs

            self.logger('Started training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

            prev_lr = np.inf
            best_loss = np.inf

            for epoch_num in range(1, n_epochs + 1):
                self.logger('===== starting Epoch # %03d' % epoch_num)

                train_avg_meter, train_loss_dict = self.train()
                self.logger("[Epoch: %d/%d] Train loss : %.5f" % (epoch_num, n_epochs, train_avg_meter.avg))
                self.logger("[Loss] j2d_loss : %.5f, total_loss : %.5f" % (
                    train_loss_dict['j2d_loss'],
                    train_loss_dict['total_loss']))

                if self.cfg.validation:
                    valid_avg_meter, valid_loss_dict, pck_meter = self.valid()
                    self.logger("[Epoch: %d/%d] Evaluation loss : %.5f" % (epoch_num, n_epochs, valid_avg_meter.avg))
                    self.logger("[Loss] j2d_loss : %.5f, total_loss : %.5f" % (
                        valid_loss_dict['j2d_loss'],
                        valid_loss_dict['total_loss']))
                    self.logger("[2D PCK] 2D PCK : %.3f"%pck_meter.avg)
                    valid_loss[epoch_num] = valid_avg_meter.avg
                    pck[epoch_num] = pck_meter.avg
                
                train_loss[epoch_num] = train_avg_meter.avg
                j2d_loss[epoch_num] = train_loss_dict['j2d_loss'].detach().cpu()
                    
                fig = plt.figure(figsize=(10, 10))
                fig.suptitle('%d epochs'%epoch_num, fontsize=16)
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.set(xlim=[0, n_epochs], title='train loss', xlabel='Epochs', ylabel='Loss')
                ax2.set(xlim=[0, n_epochs], title='keypt loss', xlabel='Epochs', ylabel='Loss')
            
                ax1.plot(train_loss.keys(), train_loss.values(), c = "r", linewidth=0.5)
                ax2.plot(j2d_loss.keys(), j2d_loss.values(), c = "r", linewidth=0.5)

                ax1.set_yscale('log')
                ax2.set_yscale('log')

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
    config_num = 0
    configs = {
        'manual_seed' : 24756,
        'ckp_dir' : '/root/sensor-fusion-gesture/ckp/Heatmap',
        'lr' : 1e-3,
        'lr_decay_gamma' : 0.5,
        'lr_decay_step' : 50,
        'lr_reduce' : False,
        'expr_ID' : 'test1',
        'cuda_id' : 0,
        'dataset' : 'FreiHAND',
        'dataset_dir' : '/root/Dataset/FreiHAND_pub_v2',
        'try_num' : 0,
        'optimizer' : "sgd",
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
        'j2d_loss_weight' : 1,
        'config_num': config_num,
        'iter' : False,
        'to_mano':None,
        'pretrained_net': False,
        # 'pretrained_net':"/root/Pretrained/S2HAND/texturehand_freihand.t7",
        'validation':True,
    } 

    config = configs
    cfg = Config(**config)
    path = os.path.join('/root/sensor-fusion-gesture/ckp/Heatmap/results%d'%config_num)
    os.makedirs(path, exist_ok=True)
    cfg.write_cfg(write_path=os.path.join(path, 'config%d.yaml'%config_num))
    model = RGB2HM()
    trainer = Trainer(cfg, model)
    trainer.fit()