import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision.transforms import ToTensor
import lmdb
from itertools import groupby

from utils.utils import preprocess, imfrombytes

class SyntheticVideoDataset(Dataset):
    """
    Dataset for synthetic videos in LMDB format. Each item is given by a window of num_input_frames input (to be
    restored) and ground-truth frames and a window of num_reference_frames reference frames.

    Args:
        data_base_path (Path): Data base path of the synthetic dataset
        num_neighbour_frames (int): Number of neigbour frames to each side
        preprocesss_mode (str): Frame preprocess mode
        crop_mode (str): Crop mode. Must be in: ["center", "random"]
        patch_size (int): Size of the croped frames

    Returns:
        dict with keys:
            "imgs_lq" (torch.Tensor): Input frames
            "imgs_gt" (torch.Tensor): Ground-truth frames
            "img_name" (str): Name of the center input frame
    """

    def __init__(self,
                 data_base_path: Path,
                 num_neighbour_frames: int = 3,
                 preprocess_mode: str = "crop",
                 crop_mode: str = "center",
                 patch_size: int = 128):
        self.data_base_path = data_base_path
        self.num_neighbour_frames = num_neighbour_frames
        self.preprocess_mode = preprocess_mode
        self.crop_mode = crop_mode
        self.patch_size = patch_size

        self.lq_base_path = data_base_path / "input"
        self.gt_base_path = data_base_path / "gt"

        self.database_lq = lmdb.open(str(self.lq_base_path / "input.lmdb"), readonly=True, lock=False, readahead=False)
        self.database_gt = lmdb.open(str(self.gt_base_path / "gt.lmdb"), readonly=True, lock=False, readahead=False)

        with open(str(self.lq_base_path / "input.lmdb" / "meta_info.txt"), 'r') as f:
            lines = f.readlines()
        # Split each line on space and take the first item
        self.img_names = [line.split()[0] for line in lines]

        # Get start and end index of each clip
        self.clip_intervals = {}
        for key, group in groupby(enumerate([str(Path(x).parent) for x in self.img_names]), key=lambda x: x[1]):
            group = list(group)
            self.clip_intervals[key] = (group[0][0], group[-1][0])

    def __getitem__(self, idx):
        center_frame_lq = self.img_names[idx]
        img_name = str(Path(center_frame_lq).name)
        clip_name = str(Path(center_frame_lq).parent)
        clip_interval_start, clip_interval_end = self.clip_intervals[clip_name]

        # Get idxs of the frames in the window
        idxs_imgs = np.arange(idx - self.num_neighbour_frames, idx + self.num_neighbour_frames + 1)
        idxs_imgs = list(idxs_imgs[(idxs_imgs >= clip_interval_start) & (idxs_imgs <= clip_interval_end)])
        imgs_lq = []

        # Get the frames from the LMDB database
        with self.database_lq.begin(write=False) as txn_lq:
            for img_idx in idxs_imgs:
                img_bytes = txn_lq.get(self.img_names[img_idx][:-4].encode('ascii'))
                img_t = ToTensor()(imfrombytes(img_bytes, float32=True))
                imgs_lq.append(img_t)

        with self.database_gt.begin(write=False) as txn_gt:
            img_bytes = txn_gt.get(self.img_names[idx][:-4].encode('ascii'))
            img_gt = ToTensor()(imfrombytes(img_bytes, float32=True))

        # Pad with black frames if the window is not complete (the center frame is too close to the start or the end of the clip)
        if len(imgs_lq) < 2 * self.num_neighbour_frames + 1:
            black_frame = torch.zeros_like(imgs_lq[0])
            missing_frames_left = self.num_neighbour_frames - (idx - clip_interval_start)
            for _ in range(missing_frames_left):
                imgs_lq.insert(0, black_frame)
            missing_frames_right = self.num_neighbour_frames - (clip_interval_end - idx)
            for _ in range(missing_frames_right):
                imgs_lq.append(black_frame)
        imgs_lq = torch.stack(imgs_lq)
        img_gt = img_gt.unsqueeze(dim=0)

        if self.preprocess_mode != "none":
            imgs_dict = preprocess({"imgs_lq": imgs_lq, "img_gt": img_gt}, mode=self.preprocess_mode,
                                                patch_size=self.patch_size, crop_mode=self.crop_mode)
            imgs_lq = imgs_dict["imgs_lq"]
            img_gt = imgs_dict["img_gt"]


        img_gt = img_gt.squeeze(dim=0)
        
        return {"imgs_lq": imgs_lq,
                "img_gt": img_gt,
                "img_name": f"{clip_name}/{img_name}"}

    def __len__(self):
        return len(self.img_names)