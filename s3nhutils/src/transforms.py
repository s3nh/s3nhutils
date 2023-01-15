import albumentations as albu
from s3nhutils.src.cfg import MainConfig
from albumentations.pytorch import ToTensorV2

#from torchvision.transforms import Compose
#from torchvision.transforms import Normalize, Resize
#from torchvision.transforms import ToTensorV2

class Augments:
    """
    Contains Train, Validation Augments 
    """
    train_augments = albu.Compose([
        albu.Resize(MainConfig.size, MainConfig.size, p=1.0), 
        albu.HorizontalFlip(p=0.5), 
        albu.VerticalFlip(p=0.5), 
        albu.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0
        ),
        ToTensorV2(p=1.0), 
    ], p =1.0)

    valid_augments = albu.Compose([
        albu.Resize(MainConfig.size, MainConfig.size, p=1.0), 
        albu.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], p=1.0)
