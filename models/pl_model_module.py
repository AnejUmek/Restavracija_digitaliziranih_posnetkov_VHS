import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics.image
import torchmetrics
from torchvision.transforms.functional import to_pil_image
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
import os.path as osp
from pathlib import Path
import json


class ModelModule(pl.LightningModule):
    """
    Pytorch Lightning Module for model training.

    Args:
        net (nn.Module): Model to train
        num_neighbour_frames (int): Number of neigbour frames to each side
        lr (float): Learning rate
        data_log_path (Path): path to data log
    """

    def __init__(self, net: nn.Module, num_neighbour_frames: int = 3, lr: float = 1e-5, data_log_path: Path = None):
        super(ModelModule, self).__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.num_neighbour_frames = num_neighbour_frames
        self.lr = lr
        self.data_log_path = data_log_path

        self.mse = torchmetrics.MeanSquaredError()

        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="alex")

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.99))

        self.losses = {
            "mse_loss": 0.0,
        }
        self.metrics = {
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 0.0
        }

        self.num_train_steps = 0
        self.num_val_steps = 0
        self.first = True

    def forward(self, *x):
        return self.net(*x)

    def training_step(self, batch, batch_idx):
        imgs_lq = batch["imgs_lq"]
        img_gt = batch["img_gt"]
        output = self.net(imgs_lq)

        mse_loss = self.mse(output, img_gt)

        log_loss = {
            "mse_loss": mse_loss
        }
        
        for loss_name in log_loss.keys():
            self.losses[loss_name] += float(log_loss[loss_name])
        
        self.num_train_steps += 1
        
        self.log_dict(log_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=img_gt.shape[0])
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        imgs_lq = batch["imgs_lq"]
        img_gt = batch["img_gt"]
        output = self.net(imgs_lq).to(torch.float32)

        output = torch.clamp(output, 0, 1)
        psnr = self.psnr(output, img_gt)
        ssim = self.ssim(output, img_gt, data_range=1.)
        with torch.no_grad():
            lpips = self.lpips(output * 2 - 1, img_gt * 2 - 1)    # Input must be in [-1, 1] range

        log_metrics = {"psnr": psnr,
                       "ssim": ssim,
                       "lpips": lpips}

        self.log_dict(log_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=img_gt.shape[0])

        for metric_name in log_metrics.keys():
            self.metrics[metric_name] += float(log_metrics[metric_name])

        self.num_val_steps += 1

        imgs_name = batch["img_name"]
        for i, img_name in enumerate(imgs_name):
            img_num = int(osp.basename(img_name)[:-4])
            if img_num % 100 == 0 or batch_idx == 0:
                single_img_gt = img_gt[0]
                single_img_output = torch.clamp(output[0], 0., 1.)
                
                single_img_lq_temp = imgs_lq[0, self.num_neighbour_frames]
                single_img_lq = torch.zeros_like(single_img_output)
                _, h, w =  single_img_lq_temp.shape
                _, h_big, w_big = single_img_lq.shape
                h_start = int((h_big - h) // 2)
                w_start = int((w_big - w) // 2)
                single_img_lq[:, h_start:h_start+h, w_start:w_start+w] = single_img_lq_temp
                
                concatenated_img = torch.cat((single_img_lq, single_img_output, single_img_gt), -1)
                self.logger.experiment.log_image(to_pil_image(concatenated_img.cpu()), str(img_num), step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        for metric_name in self.metrics.keys():
            self.metrics[metric_name] /= self.num_val_steps

        if self.data_log_path is not None and not self.first:
            with open(str(self.data_log_path), "r+") as json_file:
                data = json.load(json_file)
                for metric_name, metric_value in self.metrics.items():
                    data["validation"][metric_name].append(metric_value)
                json_file.seek(0)
                json.dump(data, json_file, indent=4)
                json_file.truncate()

        self.first = False

        for metric_name in self.metrics:
            self.metrics[metric_name] = 0.0
        self.num_val_steps = 0
        
        self.psnr.reset()
        self.lpips.reset()

    def on_train_epoch_end(self) -> None:
        for loss_name in self.losses.keys():
            self.losses[loss_name] /= self.num_train_steps

        if self.data_log_path is not None:
            with open(str(self.data_log_path), "r+") as json_file:
                data = json.load(json_file)
                for loss_name, loss_value in self.losses.items():
                    data["train"][loss_name].append(loss_value)
                json_file.seek(0)
                json.dump(data, json_file, indent=4)
                json_file.truncate()

        for loss_name in self.losses.keys():
            self.losses[loss_name] = 0.0
        self.num_train_steps = 0

    def configure_optimizers(self):
        return [self.optimizer]
