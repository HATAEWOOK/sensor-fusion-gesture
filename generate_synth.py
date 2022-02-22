import sys
sys.path.append('.')
sys.path.append('..')
import torch
from torch.utils.data import DataLoader

from datasetloader.data_loader_MSRAHT import get_dataset
from utils.train_utils import set_vis

class Generate_synthetic_hand:
    def __init__(self, dat_name, save_path, vis):
        self.dataset = get_dataset(dat_name, base_path=save_path, vis=vis)

    def generate(self):
        for idx, (sample) in enumerate(self.dataset):
