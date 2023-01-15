from abc import ABC
from abc import abstractmethod
import numpy as np
import torch
import torchvision
from PIL import Image

class BasicConfig:

    mapper: Dict = {
        'ndarray' : 'np_to_pil', 
        'Image' : 'image_mock', 
        'str' : 'str_to_pil',
    }

class MainConfig: 
    apex: bool = False
    debug: bool = False


class AbstractProcessor(ABC):

    @abstractmethod
    def load_object(self):
        ...

    @abstractmethod 
    def process_data(self):
        ...

class ImageProcessor(AbstractProcessor):

    def __init__(self, config: ):
        self._config = config
        self.mapper = self.config.mapper

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        return self._config = value

    @config.deleter
    def config(self):
        del self._config
        self._config = None

    @config.getter
    def config(self):
        return self._config

    def load_object(value: Any):
        """
        Load, return type etc. 
        """
        ...

    def _type_check(self, value: Any):
        return value.__class__.__name__

