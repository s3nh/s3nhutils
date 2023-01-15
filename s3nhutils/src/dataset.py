import albumentations as A
import numpy as np 
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Union, Any, Tuple
from src.cfg import CFG
from src.transform import get_transform
from src.transform import get_custom_transform




class CustomDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = get_transform() if transform is None else transform
        self.cfg = CFG()
        self.files = self.load_files()
        self.labels = self.cfg.labels

    def load_files(self) -> List:
        """ List all files with extension defined in CFG.extension variables 
            and its path declared in  CFG.path variables.

        Params
        --------
        None

        Returns 
        --------
        List
            list with files containes predefined extension.
        """
        return list(Path(CFG().path).rglob(f"*{CFG().extension}"))

    def load_image(self, path: str) -> np.ndarray:
        """ Load an image from configuration file path 
            and returns it as ana numpy array object.

        Params
        ------             
    
        Returns
        --------
        np.ndarray output image convert to np.ndarray format
        """
        img = Image.open(path)
        return np.array(img)

    def __len__(self) -> int:
        """
        et
        """
        return  len(self.files) 

    def __getitem__(self, ix: int) -> Tuple[Any, str]: 
        filepath: str = self.files[ix]
        img = self.load_image(path = filepath)
        if len(img.shape) > 2:
            img = img[0, :, :3]

        label = filepath.parts[-2]
        labelnum = torch.tensor(self.labels.get(label)).float()

        if self.transform is None:
           self.transform = get_transform() 

        img = self.transform(image = img)
        return img, labelnum