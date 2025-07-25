"""Training script for *Relational Knowledge Distillation*.

Example call:

python train_rkd.py \
    data_root="/path/to/av2" \
    teacher_checkpoint="/path/to/emp_teacher.ckpt" \
    student_model=emp_small \
    batch_size=128 \
    gpus=1
"""

from __future__ import annotations
import uuid
import argparse
from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.model.trainer_forecast_rkd import RKDTrainer
from src.model.trainer_forecast_rkd import Trainer as StudentTrainer

import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary, RichProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from importlib import import_module
from hydra.utils import instantiate, to_absolute_path

from src.utils.helpers import num_params

from pytorch_lightning import Callback, Trainer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    torch.use_deterministic_algorithms(True)
    torch.multiprocessing.set_start_method("spawn")
    pl.seed_everything(conf.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    output_dir = HydraConfig.get().runtime.output_dir

    datamodule = instantiate(conf.datamodule)

    log_base_dir = "/".join( conf.training.teacher.ckpt_path.split("/")[:-2] ) + "/"
    teacher_checkpoint = to_absolute_path(conf.training.teacher.ckpt_path)
    assert os.path.exists(teacher_checkpoint), f"Checkpoint {teacher_checkpoint} does not exist"
    teacher_model_path = conf.training.teacher.model.target._target_
    module = import_module(teacher_model_path[: teacher_model_path.rfind(".")])
    TeacherModel: pl.LightningModule = getattr(module, teacher_model_path[teacher_model_path.rfind(".") + 1 :])
    teacher_model = TeacherModel.load_from_checkpoint(
        teacher_checkpoint,
        decoder=conf.training.teacher.model.target.decoder
    )

    rkd_params = {
        **conf.model.target,
        **conf.training.rkd,
        "teacher_model": teacher_model,
    }
    rkd_params.pop("_target_", None) # Remove the _target_ key if it exists
    rkd_model = RKDTrainer(**rkd_params)

    ckpt_cb = ModelCheckpoint(every_n_epochs=1, save_top_k=-1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # version_name = f"version_{uuid.uuid4().hex[:6]}_rkd={conf.rkd_training}"
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=15,
        devices=4,
        strategy="ddp",
        accumulate_grad_batches=4,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        callbacks=[ckpt_cb, lr_cb],
        limit_train_batches=0.4,
        limit_val_batches=0.01, #conf.limit_val_batches,
        # default_root_dir=f"./lightning_logs/{version_name}",
    )

    trainer.fit(rkd_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
