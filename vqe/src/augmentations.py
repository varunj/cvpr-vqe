import cv2
import yaml
from omegaconf import DictConfig, OmegaConf
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
import numpy as np


def fix_transform(transform_cfg: DictConfig):
    # hack to read
    return yaml.safe_load(OmegaConf.to_yaml(transform_cfg))


class AugmentationTransform:
    def __init__(self, __version__=None, transform=None):
        if transform is None:
            self._tfm = None
        else:
            self._tfm = A.from_dict({'__version__': __version__, 'transform': fix_transform(transform)})

    def __call__(self, img):
        if self._tfm is None and self._tfm_src_fg is None:
            return img

        if self._tfm is not None:
            out = self._tfm(image=img)
            return out['image']
