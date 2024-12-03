import numpy as np
import matplotlib.pyplot as plt
import os
# Imports
from argparse import ArgumentParser
import sys
import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import glob
import optuna
import warnings
warnings.filterwarnings('ignore', message = 'probas_pred was deprecated in version 1.5 and will be removed in 1.7.Please use ``y_score`` instead.')
warnings.filterwarnings('ignore', message = 'The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.')
dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, dir)
from src.dataset import ROIDataModule
from src.models import SegmentationModel
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset import set_seed
from organize_files import delete_folder



losses = []
# Define the different training set sizes (as percentages of the full training set)
training_set_sizes = np.linspace(0.1, 1.0, 10)
if __name__ == "__main__":
    set_seed(42)
    # Parse arguments
    parser = ArgumentParser("darts")
    parser.add_argument(
        "--config_path", default="/users/ch_mariiavidmuk/shoulderai_tangent/config/seg_tangent_sign.yml", type=str
    )  # Change this argument to run other segmentations
    parser.add_argument(
        "--config_path_model", default="/users/ch_mariiavidmuk/shoulderai_tangent/config/seg_model.yml", type=str
    )  # Segmentation model config
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--encoder_name", default=None, type=str)
    parser.add_argument("--encoder_weights", default=None, type=str)

    args = parser.parse_args()

    # Open config files
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    with open(args.config_path_model, "r") as file:
        model_config = yaml.safe_load(file)

    # Add date to test folder
    # Define optional arguments
    study_name = f"dice_optimization_split_0"
    storage_name = f"sqlite:///db.sqlite3"
    # study_sum = optuna.get_all_study_summaries(storage=storage_name)
    # for i in study_sum:
    #     print(i.study_name)
            
            # Create or load the Optuna study
    study = optuna.create_study(
            study_name=study_name,
                storage=storage_name,
                direction="minimize",
                load_if_exists=True
            )

            # Perform the optimization process
    best_params = study.best_trial.params
                # Modify configuration for the best hyperparasmeters
    model_config['criterion'] = 'dice'
    model_config.update({"optimizer_name": best_params["optimizer_name"]})
    model_config.update({"decoder_use_batchnorm": best_params["decoder_use_batchnorm"]})
    model_config.update({"decoder_attention_type": best_params["decoder_attention_type"]})
    model_config.update({"encoder_name": best_params["encoder_name"]})
                
                # Step 2: Modify the data
    config["hyperpara_args"]["lr"] = best_params["lr"]
    model_config.update(config["hyperpara_args"])

    # Logger
    wandb_logger = WandbLogger(
        project=config["wandb_args"]["project"],
        name=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        version=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        offline=True,
    )
    

    num_folds = config["datamodule_args"]["k_folds"]
    just_test = model_config["just_test"]
    del model_config["just_test"]
# Train and evaluate the model for each training set size
    for size in training_set_sizes:
            # Data
        dm = ROIDataModule(fold_index = 0, train_size= size, root_dir = '/users/ch_mariiavidmuk/shoulderai_tangent/data',  **config["datamodule_args"])
        early_stopping = EarlyStopping("valid_combined_loss", patience=10, check_on_train_epoch_end=False)
        checkpoint = './check'
        os.makedirs(checkpoint, exist_ok = True)
        checkpoint_callback = ModelCheckpoint(
                    monitor = 'valid_combined_loss',
                    dirpath = checkpoint,
                    filename = f'best_checkpoint-epoch-{{epoch:02d}}_fold_{size}',
                    save_top_k = 1,
                    mode = 'min',
                )
        trainer = pl.Trainer(
                    accelerator="gpu",  # "gpu",
                    devices=1,
                    logger=wandb_logger,
                    # callbacks=[checkpoint_callback],
                    callbacks=[checkpoint_callback, early_stopping],
                    max_epochs=model_config["max_epochs"],
                    log_every_n_steps=model_config["log_every_n_steps"],
                    check_val_every_n_epoch = 1,
                    precision="16-mixed", 
                    enable_progress_bar = True
                )
        model = SegmentationModel(
                    wandb_name=config["wandb_args"]["name"],
                    mask=config["datamodule_args"]["mask"],
                    **model_config
                )
        model.set_fold(0)
        model.set_data_module(dm)
            
            
        trainer.fit(model, datamodule = dm)
            
        
        best_model_path = checkpoint_callback.best_model_path

        model_test = SegmentationModel.load_from_checkpoint(best_model_path)
        dm.setup(stage='test')
        model_test.set_data_module(dm)
        model_test.set_fold(0)
        loss = trainer.test(model_test, datamodule = dm)[0]["test_combined_loss"]
        # Record the performance metric
        losses.append(loss)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(training_set_sizes * 48, losses, marker='o')
    plt.xlabel('Training Set Size')
    plt.ylabel('Loss')
    plt.title('Model Performance vs. Training Set Size')
    plt.grid(True)
    plt.show()

    plt.savefig('performance_vs_training_set_z-score.png', dpi=300) 
    if os.path.exists("data/sagittal/ROI_slice"):
                delete_folder("data/sagittal/ROI_slice")