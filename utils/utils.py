import torch
from typing import List, Dict
from pathlib import Path
import cv2
import numpy as np
from pytorch_lightning.loggers import CometLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


def preprocess(imgs: Dict[str, torch.Tensor], mode: str = "crop", patch_size: int = 128, crop_mode: str = "center") -> Dict[str, torch.Tensor]:
    """Preprocesses a tensor of images or list of tensors of images.

    Args:
        imgs (Dict[str, torch.Tensor]): Dictionary of tensors of images.
        mode (str, optional): Preprocess mode. Values can be in ["crop", "resize"].
        patch_size (int, optional): Maximum patch size
        crop_mode (str, optional): Mode for croping. Values can be in ["center", "random"] 

    Returns:
        Dict[str, torch.Tensor], torch.Tensor]: Preprocessed images.
    """

    if mode == "crop":
        return crop(imgs, patch_size=patch_size, crop_mode=crop_mode)
    elif mode == "resize":
        return resize(imgs, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

def crop(imgs: Dict[str, torch.Tensor], patch_size: int = 128, crop_mode: str = "center") -> Dict[str, torch.Tensor]:
    """Center crops a tensor of images to patch_size.

    Args:
        img (Dict[str, torch.Tensor]): Tensor of images.
        patch_size (int, optional): Maximum patch size
        crop_mode (str, optional): Crop mode. Values can be in ["center", "random"].

    Returns:
        Dict[str, torch.Tensor]: Cropped images.
    """

    _, _, h, w = imgs["imgs_lq"].shape
    if crop_mode == "random":
        h_start = np.random.randint(0, h - patch_size)
        w_start = np.random.randint(0, w - patch_size)
    elif crop_mode == "center":
        h_start = max((h - patch_size) // 2, 0)
        w_start = max((w - patch_size) // 2, 0)
    else:
        raise ValueError(f"Unknown crop mode: {crop_mode}")
    
    d = {}
    for img_key in imgs.keys():
        _, _, h, w = imgs[img_key].shape
        if img_key == "img_gt" and h > 2 * patch_size and w > 2 * patch_size:
            d[img_key] = imgs[img_key][:, :, (2 * h_start):(2 * h_start + 2 * patch_size), (2 * w_start):(2 * w_start + 2 * patch_size)]
        elif img_key != "img_gt" and h > patch_size and w > patch_size:
            d[img_key] = imgs[img_key][:, :, h_start:(h_start + patch_size), w_start:(w_start + patch_size)]
        else:
            d[img_key] = imgs[img_key]

    return d

def resize(imgs: Dict[str, torch.Tensor], patch_size: int = 128) -> Dict[str, torch.Tensor]:
    """Resizes a tensor of images so that the biggest dimension is equal to patch_size while keeping the aspect ratio.

    Args:
        img (Dict[str, torch.Tensor]): Tensor of images.
        patch_size (int, optional): Maximum patch size

    Returns:
        Dict[str, torch.Tensor]: Resized images.
    """
    d = {}
    for img_key in imgs.keys():
        _, _, h, w = imgs[img_key].shape
        if img_key == "img_gt" and (h > 2 * patch_size or w > 2 * patch_size):
            if h > w:
                new_h = 2 * patch_size
                new_w = int(w * (2 * patch_size) / h)
            else:
                new_w = 2 * patch_size
                new_h = int(h * (2 * patch_size) / w)
            d[img_key] = (torch.nn.functional.interpolate(imgs[img_key], size=(new_h, new_w), mode="bilinear"))
        elif img_key != "img_gt" and (h > patch_size or w > patch_size):
            if h > w:
                new_h = patch_size
                new_w = int(w * patch_size / h)
            else:
                new_w = patch_size
                new_h = int(h * patch_size / w)
            d[img_key] = (torch.nn.functional.interpolate(imgs[img_key], size=(new_h, new_w), mode="bilinear"))
        else:
            d[img_key] = imgs[img_key]
    
    return d


def imfrombytes(content: bytes, flag: str = 'color', float32: bool = False) -> np.ndarray:
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def init_logger(experiment_name: str, api_key: str = None, project_name: str = None, online: bool = True) -> CometLogger:
    """
    Initializes a Comet-ML logger.

    Args:
        experiment_name (str): Experiment name.
        api_key (str, optional): Comet ML API key. Defaults to None.
        project_name (str, optional): Comet ML project name. Defaults to None.
        online (bool, optional): If True, the logger will be online. Defaults to True.
    """
    if online:
        comet_logger = CometLogger(api_key=api_key,
                                   project_name=project_name,
                                   experiment_name=experiment_name)
    else:
        comet_logger = CometLogger(save_dir="comet",
                                   project_name=project_name,
                                   experiment_name=experiment_name)
    return comet_logger
