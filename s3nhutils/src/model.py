import pytorch_lightning as pl
import timm
import torchmetrics

from abc import abstractmethod
from abc import ABC
from s3nhutils.src.cfg import MainConfig
from torch import nn
from torch import optim
from typing import List, Dict, Union
from typing import Any, TypeVar

def _check_model(name: str) -> bool:
    _all_models: List = timm.list_models()
    return name in _all_models

_loss_name: str = 'BSEWithLogitsLoss'

class BasicModel(pl.LightningModule) :

    def __init__(self, config):
        super(BasicModel, self).__init__()
        self._config = config
        self._model = None
        self.train_loss = None
        self.valid_loss = None

    def __call__(self):
        ...

    def _check_model(self, name: str) -> bool:
        _all_models: List = timm.list_models()
        return name  in _all_models

    def _create_model(self, name: str, pretrained: bool) -> None:
        _model = timm.create_model(name, pretrained = pretrained)
        return _model

    def set_features(self):
        self.fc = nn.Linear(sel.get_n_features(), self.config.n_labels)

    @property
    def model(self):
        return self._model

    def get_n_features(self):
        return self.model.head.in_features

    def _model_proces(self):
        self.model.reset_classifier(0)

    def _get_loss(loss_name: str) -> None:
        _loss = getattr(nn, name)
        return _loss

    def _get_optimizer(optim_name: str) -> None:
        _optimizer = getattr(optim, name)
        return _optimizer

    def _get_scheduler(scheduler_name: str) -> None:
        _scheduler =  getattr(torch.optim.lr_scheduler, name)
        return _scheduler

    @model.setter
    def model(self, name: str):
        self._model = self._create_model(name = name, pretrained = True)

    @model.getter
    def model(self):
        return self._model

    @model.deleter
    def model(self):
        del self._model
        self._model = None 
        
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self_config = value

    @config.getter
    def config(self):
        return self._config

    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)

    def training_step(self, batch, batch_idx: int) -> Dict:
        imgs = batch[0]
        target = batch[1]

        out = self(imgs).view(-1)
        train_loss = self.train_loss(out, target)

        logs = {'train_loss': train_loss}
        return {'loss' : train_loss, 
                'log'  : logs
        }

    def validation_step(self, batch, batch_idx: int) -> Dict:
        ...

    def configure_optimizers(self):
        ...