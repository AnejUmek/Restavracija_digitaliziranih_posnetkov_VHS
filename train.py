import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.strategies import DDPStrategy
from dotmap import DotMap
import json
import os

from data.synthetic_video_dataset import SyntheticVideoDataset
from data.pl_video_data_module import VideoDataModule
from models.pl_model_module import ModelModule
from utils.utils import PROJECT_ROOT
from utils.generate_graphs import generate_graphs_train, generate_graphs_validation

def train(args: DotMap, net: nn.Module, logger: CometLogger):
    """
    Train the model on the synthetic dataset.

    Args:
        args: Arguments
        net: Network to be trained
        logger: Comet ML logger
    """

    # Set seed
    pl.seed_everything(27, workers=True)
    os.environ['PYTHONHASHSEED'] = str(27)

    data_base_path = Path(args.data_base_path)
    checkpoints_path = PROJECT_ROOT / "experiments" / args.experiment_name / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=False)
    with open(data_base_path / "config.json", "w") as f:
        json.dump(args, f, indent=4)

    training_path = data_base_path / "train"
    val_path = data_base_path / "validation"
    if not val_path.exists():
        val_path = data_base_path / "test"

    data_log_path = PROJECT_ROOT / "experiments" / args.experiment_name / "data_log.json"

    if data_log_path is not None:
        with open(str(data_log_path), "r+") as json_file:
            data = json.load(json_file)
            data["train"] = {
                "mse_loss": []
            }
            data["validation"] = {
                "psnr": [],
                "ssim": [],
                "lpips": []
            }
            json_file.seek(0)
            json.dump(data, json_file, indent=4)
            json_file.truncate()

    train_dataset = SyntheticVideoDataset(data_base_path=training_path,
                                          num_neighbour_frames=args.num_neighbour_frames,
                                          preprocess_mode="crop",
                                          crop_mode="random",
                                          patch_size=args.train_patch_size)

    val_dataset = SyntheticVideoDataset(data_base_path=val_path,
                                        num_neighbour_frames=args.num_neighbour_frames,
                                        preprocess_mode="crop",
                                        crop_mode="center",
                                        patch_size=args.train_patch_size)

    data_module = VideoDataModule(train_dataset, val_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path,
                                          filename="{epoch}-{step}-{lpips:.3f}",
                                          save_weights_only=True,
                                          monitor="lpips",
                                          save_top_k=1,
                                          save_last=True)
    checkpoint_callback.FILE_EXTENSION = ".pth"

    model = ModelModule(net=net,
                        num_neighbour_frames=args.num_neighbour_frames,
                        lr=args.lr,
                        data_log_path=data_log_path)

    trainer = Trainer(**args.training_params, strategy=DDPStrategy(find_unused_parameters=True, static_graph=True),
                      logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    if args.generate_graphs:
        generate_graphs_train(data_log_path)
        generate_graphs_validation(data_log_path)
