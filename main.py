import albumentations as albu
import cv2
import os 
import pathlib
import pandas as pd
import timm
import torch
import numpy as np
import logging
from rich.logging import RichHandler
from torch.utils.data import Dataset
from torch import nn

from pathlib  import Path
from typing import List, Dict, Union
from typing import Any, TypeVar
from PIL import Image #To check what ll be faster 

from model import CustomModel
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import  CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")
# ====================================================
# scheduler 
# ====================================================
def get_scheduler(optimizer):
    if CFG.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
    elif CFG.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    return scheduler


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BreastDataset(Dataset):
    def __init__(self, df, transforms=None):
        """
        Data generator 
        """
        self.df = df
        self.transforms = self._basic_transform()
        self.paths = df['path'].values
        self.targets = df['cancer'].values
    
    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        self._transforms = value

    @transforms.getter
    def transforms(self):
        return self._transforms

    @transforms.deleter
    def transforms(self):
        del self._transforms

    def _basic_transform(self):
        _transform = [ 
            albu.Normalize(mean = 0.0, std = 1.0), 
            ToTensorV2(),
            ] 
        return albu.Compose(_transform, p = 1)
    
    def _check_files(self):
        assert len(self.paths) > 0, 'Data should not be empty'

    def _create_weight(self) -> torch.tensor:
        return torch.tensor([1])


    def _load_image(path: Union[str, pathlib.Path]) -> torch.tensor:
        image = cv2.imread(path)
        if self.transforms is not None:
            image = self.transforms(image = image)["image"]
            

    def __len__(self):
        return len(self.paths)

    def read_label(self, value: int)   -> torch.Tensor:
        return torch.tensor([value], dtype = torch.float)        

    def __getitem__(self, idx: int):
        """
        Item accessor  
        """
        image = Image.open(self.paths[idx]).convert('RGB')
        image = np.array(image)
        if self.transforms is not None:
            image = self.transforms(image = image)["image"]
            
        #It just does not work really well.
        #y = self.read_label(value = self.targets[idx]) 
        y = self.targets[idx]
        w = self._create_weight()
        return image, y

def main():

    Pathable = Union[str, pathlib.Path]
    PATH: Pathable = '/root/data/'
    TABPATH: Pathable = '/root/data/train.csv'

    df = pd.read_csv(TABPATH)
    bd = BreastDataset(df = df)
    log.info(f"Number of obs in dataset {len(bd)}")    
    model = timm.create_model('vit_large_patch32_224', img_size = 512, num_classes =1,  pretrained = False)

    train_loader = DataLoader(bd, batch_size = 12, 
                          shuffle = True, 
                          num_workers = 4, pin_memory = True, 
                          drop_last = True)


    losses = AverageMeter()
    scores = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    device = 'cuda'
    criterion = nn.BCEWithLogitsLoss()
    model = model.cuda()
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        log.info(f"{image.shape}")
        labels = labels.cuda()
        log.info(f"{labels.shape}")
        batch_size = 12
        y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    main()