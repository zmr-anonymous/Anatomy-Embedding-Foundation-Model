import time
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from trainer.trainer_general import trainer_general
from utility import *

class trainer_segmentation(trainer_general):
    '''
    basic trainer for segmentation
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        super(trainer_segmentation, self).__init__(config)

        assert config['Task']['task_type'] == 'segmentatin', 'wrong task type.'
        self.targets = config['Task']['seg_targets']

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
        self.best_metric = 0

    def generate_dataloader(self):
        self.train_loader = self.dataloader.get_train_loader()
        self.val_loader = self.dataloader.get_val_loader()
        self.post_trans = self.dataloader.get_post_transforms()

    def inference(self, input):
        '''
        method to inference data with current model
        '''
        def _foreword(input):
            # global_feature, local_feature = self.sam_network(input)
            # outputs = self.network(input, local_feature)
            outputs = self.network(input)
            return outputs

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.dataloader.get_roi_size(),
                sw_batch_size=1,
                predictor=_foreword,
                overlap=0.5,
                device=torch.device('cpu')
            )
        if self.val_amp:
            with torch.cuda.amp.autocast():
                val_outputs = _compute(input)
        else:
            val_outputs = _compute(input)

        return val_outputs
    
    def validation_eval(self, training_metric_dict):
        self.network.eval()
        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(self.device),
                    val_data["label"].to('cpu'),
                )
                val_outputs = self.inference(val_inputs)
                # val_outputs_bin = torch.where(val_outputs > 0.5, torch.ones_like(val_outputs), torch.zeros_like(val_outputs))
                self.dice_metric(y_pred=val_outputs, y=val_labels)
                self.dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = self.dice_metric.aggregate().item()

            training_metric_dict['val_dice'] = metric
            metric_batch = self.dice_metric_batch.aggregate()
            for i, target in enumerate(self.targets):
                training_metric_dict['val_dice_'+target] =  metric_batch[i].item()

            self.dice_metric.reset()
            self.dice_metric_batch.reset()

            if metric > self.best_metric:
                best_metric_dict = {'epoch':self.epoch + 1}
                self.best_metric = metric
                best_metric_dict['dice'] = self.best_metric
                self.metric_log['best_metric_list'].append(best_metric_dict)

                self.save_checkpoint("best_val_metric_model.pth")
                print_to_log_file("saved new best metric checkpoint")
            print_to_log_file(f"current epoch: {self.epoch + 1} current mean dice: {metric:.4f}")
            log_to_print = ''
            for i, target in enumerate(self.targets):
                log_to_print = log_to_print + " " + target + ": " + f"{metric_batch[i].item():.4f}"
            print_to_log_file(log_to_print)
            print_to_log_file(    
                f"best mean dice: {self.best_metric:.4f}"
                f" at epoch: {self.epoch + 1}"
            )

    def train_epoch(self, scaler):
        self.network.train()
        epoch_loss = 0
        step = 0
        for batch_data in self.train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(globalVal.device),
                batch_data["label"].to(globalVal.device),
            )
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.network(inputs)
                loss = self.loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            epoch_loss += loss.item()
        self.lr_scheduler.step()
        epoch_loss /= step
        return epoch_loss

