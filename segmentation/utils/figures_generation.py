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
from src.organize_files import delete_folder


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
    parser.add_argument("--hyper_param", default=False, type=str) # If you want to load hyperparameters from database 
    args = parser.parse_args()

    # Open config files
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    with open(args.config_path_model, "r") as file:
        config_model = yaml.safe_load(file)

    # Add date to test folder
    date = datetime.now().strftime('%Y-%m-%d')
    res_folder = config_model["test_results_folder"]
    res_folder_date = f"{date}_{res_folder}"
    config_model["test_results_folder"] = res_folder_date
    # Define optionals arguments
    if args.model_name is not None:
        config_model["model_name"] = args.model_name
    if args.encoder_name is not None:
        config_model["encoder_name"] = args.encoder_name
    if args.encoder_weights is not None:
        config_model["encoder_weights"] = args.encoder_weights
    if args.hyper_param: 
        study_name = f"dice_optimization_split_0"
        storage_name = f"sqlite:///db.sqlite3"
        study = optuna.create_study(
            study_name=study_name,
                storage=storage_name,
                direction="minimize",
                load_if_exists=True
            )

            # Perform the optimization process
        best_params = study.best_trial.params
                        # Modify configuration for the best hyperparasmeters
        config_model['criterion'] = 'dice'
        config_model.update({"optimizer_name": best_params["optimizer_name"]})
        config_model.update({"decoder_use_batchnorm": best_params["decoder_use_batchnorm"]})
        config_model.update({"decoder_attention_type": best_params["decoder_attention_type"]})
        config_model.update({"encoder_name": best_params["encoder_name"]})
                        
                        # Step 2: Modify the data
        config["hyperpara_args"]["lr"] = best_params["lr"]
    config_model.update(config["hyperpara_args"])
    # Logger
    wandb_logger = WandbLogger(
        project=config["wandb_args"]["project"],
        name=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        version=config["wandb_args"]["name"] + config["wandb_args"]["version"],
        offline=True,
    )
    

    num_folds = config["datamodule_args"]["k_folds"]
    just_test = config_model["just_test"]
    del config_model["just_test"]
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
                    filename = f'best_checkpoint-epoch-{{epoch:02d}}_fold_{0}',
                    save_top_k = 1,
                    mode = 'min',
                )
        trainer = pl.Trainer(
                    accelerator="gpu",  
                    devices=1,
                    logger=wandb_logger,
                    callbacks=[checkpoint_callback, early_stopping],
                    max_epochs=config_model["max_epochs"],
                    log_every_n_steps=config_model["log_every_n_steps"],
                    check_val_every_n_epoch = 1,
                    precision="16-mixed", 
                    enable_progress_bar = True
                )
        model = SegmentationModel(
                    wandb_name=config["wandb_args"]["name"],
                    mask=config["datamodule_args"]["mask"],
                    **config_model
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

    loss = [4.8293, 3.7285,  3.5486, 3.4321, 3.12, 2.6298,  2.4591, 2.2972,  2.017, 1.3757]
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(training_set_sizes * 62, loss, marker='o')
    plt.xlabel('Training Set Size')
    plt.ylabel('Loss')
    plt.title('Model Performance vs. Training Set Size')
    plt.grid(True)
    plt.show()

    plt.savefig('performance_vs_training_z_score.png', dpi=300) 
    if os.path.exists("data/sagittal/ROI_slice"):
                delete_folder("data/sagittal/ROI_slice")


def plot_val_losses(all_fold_losses):
# Assuming all_fold_losses is already defined and populated with data

# Create a figure with 10 subplots (2 rows, 5 columns)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.ravel()  # Flatten the axes array

    # Plot each fold's validation loss in its own subplot
    for i, fold_losses in enumerate(all_fold_losses):
        axs[i].plot(fold_losses)
        axs[i].set_title(f"Fold {i+1}")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Validation Loss")

    plt.tight_layout()  # Adjust subplots to fit into figure

    # Save the figure as an image file
    plt.savefig("validation_loss_folds.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

    plt.show()  # Show the plot
