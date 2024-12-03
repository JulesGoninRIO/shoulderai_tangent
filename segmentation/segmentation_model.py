# Imports
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import warnings
import numpy as np
from src.dataset import ROIDataModule
from src.models import SegmentationModel
from utils.utils_loss import add_loss_the_foldername
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset import set_seed
from src.organize_files import compute_average_loss
from utils.figures_generation import plot_val_losses

# Suppress specific warnings
warnings.filterwarnings('ignore', message='probas_pred was deprecated in version 1.5 and will be removed in 1.7. Please use ``y_score`` instead.')
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.')

def train_model(config, config_model, args):
    set_seed(42)

    # Logger setup
    wandb_logger = WandbLogger(
        project=config["wandb_args"]["project"],
        name=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        version=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        offline=True,
    )

    # Add date to results folder
    date = datetime.now().strftime('%Y-%m-%d')
    res_folder = config_model["test_results_folder"]
    res_folder_date = f"{date}_{res_folder}"
    config_model["test_results_folder"] = res_folder_date

    num_folds = config["datamodule_args"]["k_folds"]
    all_fold_losses = []

    for fold_index in range(num_folds):
        print(f"Training Fold {fold_index}")
        dm = ROIDataModule(fold_index=fold_index, **config["datamodule_args"])
        early_stopping = EarlyStopping("valid_combined_loss", patience=10, check_on_train_epoch_end=False)
        checkpoint = './check'
        os.makedirs(checkpoint, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_combined_loss',
            dirpath=checkpoint,
            filename=f'best_checkpoint-epoch-{{epoch:02d}}_fold_{fold_index}',
            save_top_k=1,
            mode='min',
        )
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
            max_epochs=config_model["max_epochs"],
            log_every_n_steps=config_model["log_every_n_steps"],
            check_val_every_n_epoch=1,
            precision="16-mixed", 
            enable_progress_bar=True, 
            gradient_clip_val=1.0
        )

        model = SegmentationModel(
            wandb_name=config["wandb_args"]["name"],
            mask=config["datamodule_args"]["mask"],
            lr = config["hyperpara_args"]["lr"],
            beta1 = config["hyperpara_args"]["beta1"],
            beta2 = config["hyperpara_args"]["beta2"], 
            **config_model
        )
        model.set_fold(fold_index)
        model.set_data_module(dm)
        trainer.fit(model, datamodule=dm)
        all_fold_losses.append(model.validation_losses)
    
    plot_val_losses(all_fold_losses)
    wandb.finish()

def test_model(config, config_model, args):
    set_seed(42)
    checkpoint = './check'
    # Logger setup
    wandb_logger = WandbLogger(
        project=config["wandb_args"]["project"],
        name=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        version=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        offline=True,
    )

    # Add date to results folder
    date = datetime.now().strftime('%Y-%m-%d')
    res_folder = config_model["test_results_folder"]
    res_folder_date = f"{date}_{res_folder}"
    config_model["test_results_folder"] = res_folder_date

    num_folds = config["datamodule_args"]["k_folds"]
    test_losses_folds = []

    for fold_index in range(num_folds):
        print(f"Testing Fold {fold_index}")
        dm = ROIDataModule(fold_index=fold_index, **config["datamodule_args"])

        # Find the checkpoint file for the fold
        checkpoint_suffix = f"_fold_{fold_index}.ckpt"
        checkpoint_files = [
            f for f in os.listdir('checkpoint')
            if f.endswith(checkpoint_suffix)
        ]
        if len(checkpoint_files) == 0:
            raise FileNotFoundError(f"No checkpoint found for fold {fold_index}")
        elif len(checkpoint_files) > 1:
            raise ValueError(f"Multiple checkpoints found for fold {fold_index}, expected one: {checkpoint_files}")

        best_model_path = os.path.join(checkpoint, checkpoint_files[0])
        print(f"Best model path for fold {fold_index}: {best_model_path}")

        model_test = SegmentationModel.load_from_checkpoint(best_model_path)
        model_test.test_results_folder = res_folder_date + config["wandb_args"]["name"]
        dm.setup(stage='test')
        model_test.set_data_module(dm)
        model_test.set_fold(fold_index)
        print(f"Dataset: {dm.test_dataset}")
        checkpoint = './check'
        checkpoint_callback = ModelCheckpoint(
                monitor = 'valid_combined_loss',
                dirpath = checkpoint,
                filename = f'best_checkpoint-epoch-{{epoch:02d}}_fold_{fold_index}',
                save_top_k = 1,
                mode = 'min',
            )
        early_stopping = EarlyStopping("valid_combined_loss", patience=10, check_on_train_epoch_end=False)

        trainer = pl.Trainer(
                accelerator="gpu", 
                devices=1,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, early_stopping],
                max_epochs=config_model["max_epochs"],
                log_every_n_steps=config_model["log_every_n_steps"],
                check_val_every_n_epoch = 1,
                precision="16-mixed", 
                enable_progress_bar = True, 
                gradient_clip_val=1.0
            )

        test_results = trainer.test(model_test, datamodule=dm)[0]
        test_losses_folds.append(test_results["test_combined_loss"])

    compute_average_loss(os.path.join(res_folder_date, config["wandb_args"]["name"]))
    add_loss_the_foldername(res_folder_date, np.mean(test_losses_folds))
    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--config_path", default="./config/seg_tangent_sign.yml", type=str)
    parser.add_argument("--config_path_model", default="./config/seg_model.yml", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--encoder_name", default=None, type=str)
    parser.add_argument("--encoder_weights", default=None, type=str)

    args = parser.parse_args()

    # Load configuration files
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    with open(args.config_path_model, "r") as file:
        config_model = yaml.safe_load(file)

    # Update optional arguments
    if args.model_name:
        config_model["model_name"] = args.model_name
    if args.encoder_name:
        config_model["encoder_name"] = args.encoder_name
    if args.encoder_weights:
        config_model["encoder_weights"] = args.encoder_weights

    if args.mode == "train":
        train_model(config, config_model, args)
    elif args.mode == "test":
        test_model(config, config_model, args)
