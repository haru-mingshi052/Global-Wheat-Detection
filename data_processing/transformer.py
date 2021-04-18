import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

"""
transformer
"""

def train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p = 1.0)
    ], bbox_params = {'format':'pascal_voc', 'label_fields':['labels']})

def val_transform():
    return A.Compose([
        ToTensorV2(p = 1.0)
    ], bbox_params = {'format':'pascal_voc', 'label_fields':['labels']})

def test_transform():
    return A.Compose([
        ToTensorV2(p = 1.0)
    ])