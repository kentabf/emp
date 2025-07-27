"""Training script for *Relational Knowledge Distillation*.

"""

from __future__ import annotations
import uuid
import argparse
from pathlib import Path
import datetime
import torch.profiler as profiler
from pytorch_lightning.profilers import SimpleProfiler

import torch
import pickle
import json

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

    from src.utils.helpers import num_params
    print(f"Number of parameters in RKD {conf.model.name} model: {num_params(rkd_model) - num_params(teacher_model)}")

    ckpt_cb = ModelCheckpoint(every_n_epochs=1, save_top_k=-1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{run_time}_{conf.model.name}_rkd={conf.training.rkd.rkd_on}"
    profiler = SimpleProfiler()
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=60, 
        devices=1,
        # strategy="ddp",
        # accumulate_grad_batches=4,
        profiler=profiler,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        callbacks=[ckpt_cb, lr_cb],
        # limit_train_batches=0.005,
        # limit_val_batches=0.005, #conf.limit_val_batches,
        default_root_dir=f"./lightning_logs/{version_name}",
    )

    trainer.fit(rkd_model, datamodule=datamodule)

    epoch_times = profiler.recorded_durations["run_training_epoch"]
    print("Wall clock time per epoch:", [round(t, 2) for t in epoch_times])
    # save wall clock data as JSON
    json_path = f"./lightning_logs/{version_name}/epoch_times.json"
    with open(json_path, "w") as f:
        json.dump(list(epoch_times), f)
    # pickle the profiler
    pickle_path = f"./lightning_logs/{version_name}/profiler.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(profiler, f)


if __name__ == "__main__":
    main()
