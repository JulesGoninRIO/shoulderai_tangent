# Imports
import os

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import precision_recall_curve, mean_squared_error
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.loss import TangentSignLoss, dice_coefficient 
from utils.utils_preproc import*
from src.dataset import set_seed, ROIDataset
from src.organize_files import save_array_to_csv


class SegmentationModel(pl.LightningModule):
    """Segmentation model definition."""

    def __init__(
        self,
        max_epochs,
        model_name,
        encoder_name,
        encoder_depth,
        encoder_weights,
        in_channels,
        num_classes,
        criterion,
        optimizer_name,
        lr,
        beta1, 
        beta2,
        threshold_pred,
        weight_decay,
        log_every_n_steps,
        wandb_name,
        mask,
        test_results_folder,
        batch_size, 
        decoder_attention_type = None

    ):
        super().__init__()
        self.save_hyperparameters()
        set_seed(42)
        self.initial_loss = 100
        self.max_epochs = max_epochs
        self.optimizer_name = optimizer_name
        self.threshold_pred = threshold_pred
        self.lr = lr
        self.beta1 = beta1 
        self.beta2 = beta2
        self.fold = 0
        self.weight_decay = weight_decay
        self.log_every_n_steps = log_every_n_steps
        self.mask = mask
        self.criterion = criterion
        self.epoch_outputs = [] 
        self.validation_outputs = []
        self.first_epoch_completed = False
        self.dice_coeff_train = torchmetrics.Dice(
            zero_division=1, num_classes=1, multiclass=False, average="samples"
        )
        self.dice_coeff_valid = torchmetrics.Dice(
            zero_division=1, num_classes=1, multiclass=False, average="samples"
        )

        self.test_results_folder = test_results_folder + wandb_name
        os.makedirs(self.test_results_folder, exist_ok=True)
        self.data_module = None
        self.batch_size = batch_size
        if model_name == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                decoder_channels=(256, 128, 64, 32, 16),
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                decoder_channels=(256, 128, 64, 32, 16),
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                decoder_attention_type=decoder_attention_type
            )
        elif model_name == "manet":
            self.model = smp.MAnet(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                decoder_channels=(256, 128, 64, 32, 16),
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "linknet":
            self.model = smp.Linknet(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "pspnet":
            self.model = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "pan":
            self.model = smp.PAN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "deeplabv3":
            self.model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        elif model_name == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if self.criterion == "dice":
            self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True, log_loss = True)
            self.dice_loss_test = smp.losses.DiceLoss(mode="binary", from_logits=True, log_loss = False)

            self.bce_loss = nn.BCEWithLogitsLoss()
        elif self.criterion == "dice+mse" and self.mask == "tangent_sign":
            self.loss_tangent_sign = TangentSignLoss(angle_loss=1, dist_penalty=1.0, mse_penalty=2.0)
            self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        elif self.criterion == "mse" and self.mask == "tangent_sign":
            self.loss_tangent_sign = TangentSignLoss(angle_loss=1, dist_penalty=1.0, mse_penalty=2.0)
            self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        else:
            raise NotImplementedError
        self.validation_losses = []  # Store validation losses
        self.predictor = lambda x: (x.sigmoid() > self.threshold_pred).float()


    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  betas=(self.beta1, self.beta2))
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                nesterov=True,
            )
        elif self.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.99)  # You can adjust alpha if needed
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.optimizer_name}")

        # lr_scheduler definition
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "valid_combined_loss",
                "interval": "epoch",
            },
        }

    def forward(self, img):
        torch.cuda.empty_cache()  
        mask = self.model(img)
        self.epoch_outputs = [] 
        return mask
    
    def set_data_module(self, data_module):
        self.data_module = data_module

    
    def set_data_module_add(self, data_module):
        self.data_module_add = data_module
    
    

    def shared_step(self, stage, batch, batch_idx):
        [img, mask, img_visible] = batch.values()
        mask = mask.unsqueeze(1)
        # img_visible = img.clone()
        img = z_score_normalize_batch(img)
        if self.mask == "tangent_sign" and self.criterion == "custom":
            self.loss_tangent_sign.reset()
        logits = self.forward(img)
        dice_loss = self.dice_loss(logits, mask)
        probas_mask = logits.sigmoid()
        probas_mask = torch.nan_to_num(probas_mask, nan= 0.0)
        precision, recall, thresholds = precision_recall_curve(
            y_true=mask.cpu().detach().numpy().flatten().astype(float),
            probas_pred=probas_mask.cpu().detach().numpy().flatten().astype(float),
            pos_label=1,
        )
    
        fscore = (2 * precision * recall) / (precision + recall + 1e-6)
        index = np.argmax(fscore)
        threshold = round(thresholds[index], ndigits=4)
        preds = (probas_mask > threshold).float()

        if self.mask == "tangent_sign":
            preds_thin = np.array([skeletonize_mask(pred.cpu().numpy()) for pred in preds])
            preds_tensor = torch.from_numpy(preds_thin).to(device=self.device, dtype=torch.float32)
            
        else:
            preds_tensor = preds

        if self.mask == "tangent_sign" and self.criterion == "dice+mse":
            combined_loss = dice_loss + F.mse_loss(logits.float(), mask.float())
        elif self.mask == "tangent_sign" and self.criterion == "dice":
            combined_loss = dice_loss + self.bce_loss(logits.float(), mask.float())
        elif self.mask == "tangent_sign" and self.criterion == "mse":
            combined_loss = F.mse_loss(logits.float(), mask.float())
        else:
            raise Exception("No such criterion")
        if torch.isnan(combined_loss):
            combined_loss = torch.tensor(1000)
        self.log(
                "{}_combined_loss".format(stage),
                combined_loss.item(),
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        if stage == "test":
            for i in range(len(img)):
                img_visib = img_visible[i].cpu().numpy()
                pred_np_dilated = preds[i,0].cpu().numpy()
                pred_np_filtered = filter_segments(pred_np_dilated, size_threshold=10, distance_threshold=100)
                pred_np = skeletonize_mask(pred_np_filtered)
                mask_np = mask[i, 0].cpu().numpy()
                correct_idx = batch_idx*self.batch_size + i
                patient_name = self.data_module.test_dataset.get_patient(correct_idx)
                test_results_dir = os.path.join(self.test_results_folder, f'test_results_fold_{self.fold}')
                os.makedirs(test_results_dir, exist_ok=True)
                difference = None 
                if self.mask == "tangent_sign":
                    loss = dice_coefficient(pred_np_filtered, mask_np)
                    patient_name = patient_name[:-4]
                    img_path_mask = os.path.join(test_results_dir, f"{patient_name}_{loss:.2f}_mask_thick.jpg")
                    save_overlay_image_mask(img_visib, mask_np, img_path_mask)
                    pred_np_proces = fit_line(pred_np)
                    mask_np = thin_lines(mask_np)
                    mask_np_cont = fit_line(mask_np, extend = False)
                    img_path_mask = os.path.join(test_results_dir, f"{patient_name}_{loss:.2f}_mask.jpg")
                    save_overlay_image_mask(img_visib, mask_np_cont, img_path_mask)
                    data_supra = ROIDataset(
                        data_dir=self.data_module.data_dir,
                        img_type=self.data_module.img_type,
                        mask="supraspinatus",
                        phase="train",
                        split_labels_path=None,
                        transform=self.data_module.train_transform,
                        thicken_masks=False,
                    )
                    idx = data_supra.get_index_by_patient_id(patient_name)
                    supra_mask = None
                    dice_mucle = None 
                    if idx and not patient_name.endswith("aug"):
                        print("Patient", patient_name)
                        supra_mask = data_supra[idx]["mask"]
                        difference = np.abs(calculate_percent_below(supra_mask, pred_np_proces, img_visib)- calculate_percent_below(supra_mask, mask_np_cont, img_visib))
                        muscle_below_pred = segment_below_line(supra_mask, pred_np_proces)
                        muscle_below_mask = segment_below_line(supra_mask, mask_np_cont)
                        dice_muscle = dice_coefficient(muscle_below_pred, muscle_below_mask)
                        print("Difference in percentage below the line", difference)
                    else:
                        print("No supraspinatus annotation for", patient_name)
                if idx and not patient_name.endswith("aug"):
                    between = segment_muscle_between_lines(supra_mask, mask_np_cont, pred_np_proces)
                    overlay_path = os.path.join(test_results_dir, f"{patient_name}_{loss:.2f}_loss_test_ov.jpg")
                    save_overlay_image(img_visib, mask_np_cont, segment_below_line(supra_mask, mask_np_cont), overlay_path, color_pred = 2)
                    overlay_path = os.path.join(test_results_dir, f"{patient_name}_{loss:.2f}_loss_test_o.jpg")
                    save_overlay_image(img_visib, pred_np_proces, segment_below_line(supra_mask, pred_np_proces), overlay_path)
                    overlay_path = os.path.join(test_results_dir, f"{patient_name}_{loss:.2f}_loss_test.jpg")
                    save_overlay_3_mask(img_visib, pred_np_proces, mask_np_cont, between, overlay_path)
                    overlay_path = os.path.join(test_results_dir, f"{patient_name}_{loss:.2f}_loss_pred_mask.jpg")
                    save_overlay_3_mask(img_visib, pred_np_proces, mask_np_cont, np.zeros([512, 512]), overlay_path)
                if not patient_name.endswith("aug"):
                    if difference:
                        save_array_to_csv([patient_name, round(dice_muscle.item(), 4), difference], os.path.join(self.test_results_folder, patient_name + "_losses.csv"))
                   


        torch.cuda.empty_cache()  
        return {"loss": combined_loss, "threshold": threshold, "preds": preds_tensor, "dice_loss": dice_loss}
   
    
    
    def on_train_epoch_start(self):
        torch.cuda.empty_cache()  
        self.epoch_outputs = [] 

    def training_step(self, batch, batch_idx):
        result = self.shared_step("train", batch, batch_idx)
        self.epoch_outputs.append(result)
        torch.cuda.empty_cache()  
        return result

    def on_train_epoch_end(self):
        if self.epoch_outputs:
            mean_threshold = sum(x["threshold"] for x in self.epoch_outputs) / len(self.epoch_outputs)
            self.threshold_pred = mean_threshold
            self.log('mean_threshold', self.threshold_pred)
            torch.cuda.empty_cache()
        self.epoch_outputs = []  # Reset pour la prochaine époque
        
    def set_fold(self,fold):
        self.fold = fold

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache() 
        outputs =  self.shared_step("valid", batch, batch_idx)
        start_idx = batch_idx*len(batch["image"])
        end_idx = start_idx + len(batch["image"])
        img_indices = self.data_module.val_dataset.indices[start_idx:end_idx]
        img_names = [self.data_module.val_dataset.dataset.get_patient(idx) for idx in img_indices]
        imgs = batch["image"]
        preds = outputs["preds"]

        losses = [outputs["loss"]]*len(img_names)

        results = {
            "img_names" : img_names,
            "imgs": imgs,
            "preds": preds,
            "loss": losses
        }
        self.validation_outputs.append(results)
        return results

        
    def save_overlay_image(self, img, pred, output_path):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img[0]

        pred = (pred * 255).astype(np.uint8)

        overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        overlay[..., 0] = pred  
        overlay[..., 1] = pred  
        overlay[..., 2] = 0     

        overlay = cv2.addWeighted(overlay, 0.5, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imwrite(output_path, overlay)

    def on_validation_epoch_end(self):
        all_img_names = []
        all_imgs = []
        all_preds = []
        all_losses = []

        images_seen = set()

        for output in self.validation_outputs:
            
            
            all_img_names.extend(output["img_names"])
            all_imgs.extend(output["imgs"])
            all_preds.extend(output["preds"])
            all_losses.extend(output["loss"])

        self.validation_outputs = []

        val_loss = self.trainer.callback_metrics["valid_combined_loss"]
        self.validation_losses.append(val_loss.cpu().numpy())


    def test_step(self, batch, batch_idx):
        result = self.shared_step("test", batch, batch_idx)
        self.epoch_outputs.append(result)
        torch.cuda.empty_cache()  
        return result
    

    def predict_step(self, batch, batch_idx):
        img, _ = batch
        mask = self(img)
        pred = self.predictor(mask)
        torch.cuda.empty_cache()  
        return pred
    

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def from_checkpoint(checkpoint_path, **kwargs):
        model = SegmentationModel(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        return model


