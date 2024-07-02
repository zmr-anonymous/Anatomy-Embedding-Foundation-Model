import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from network.SAM.sam_unet_3d import sam_unet_3d
from network.SAM.sam_unet_3d_2 import sam_unet_3d_2
from network.SAM.sam_with_fullconn_head import sam_with_fullconn_head
from network.SAM.sam_with_unet import sam_with_unet
from network.SAM.sam_with_cluster import sam_with_cluster
from trainer.trainer_segmentation import trainer_segmentation
from utility import *

class trainer_SAM_downstream_seg(trainer_segmentation):
    '''
    general trainer
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        if not hasattr(self, 'trainer_config'):
            self.trainer_config = config['Train']['trainer_SAM_downstream_seg']

        self.sam_target_feature_size = self.trainer_config['sam_target_feature_size']
        self.sam_num_res_units = self.trainer_config['sam_num_res_units']
        self.checkpoint_name = self.trainer_config['sam_model_name']

        self.input_dropout = self.trainer_config['input_dropout']
        self.sam_network_name = self.trainer_config['network_name']

        super(trainer_SAM_downstream_seg, self).__init__(config)

        self.disabel_sam = self.model.get_disable_sam()
        if not self.disabel_sam:
            assert isfile(self.checkpoint_name), "error: checkpoint file does not exist. from trainer_SAM_downstream_seg.py"
            checkpoint = torch.load(self.checkpoint_name)

            if self.sam_network_name == 'sam_with_unet':
                self.sam_network = sam_with_unet(
                    sam_channels=self.sam_target_feature_size,
                    out_channels=14,
                ).to(globalVal.device)

            if self.sam_network_name == 'sam_unet_3d' or self.sam_network_name == 'sam_unet_3d_2':
                self.sam_network.load_state_dict(checkpoint['net'])
            else:
                # self.sam_network.load_state_dict({k.replace('sam.', ''): v for k, v in checkpoint['net'].items()})
                filtered_dict = {key.replace('sam.', ''): value for key, value in checkpoint['net'].items() if 'sam.' in key}
                self.sam_network.sam.load_state_dict(filtered_dict)
            self.sam_network.eval()
        
        self.roi_size = self.dataloader.get_roi_size()
    
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
                if not self.disabel_sam:
                    if self.sam_network_name == 'sam_unet_3d':
                        global_feature, local_feature = self.sam_network(inputs)
                        local_feature = local_feature.detach()
                        global_feature = global_feature.detach()

                        if self.input_dropout:
                            r = torch.rand(1)
                            if r[0] > 0.5:
                                if r[0] > 0.75:
                                    inputs = torch.zeros_like(inputs)
                                else:
                                    global_feature = torch.zeros_like(global_feature)
                                    local_feature = torch.zeros_like(local_feature)
                        outputs = self.network(inputs, global_feature, local_feature)
                    elif self.sam_network_name == 'sam_unet_3d_2':
                        local_feature, _ = self.sam_network(inputs)
                        local_feature = local_feature.detach()
                        if self.input_dropout:
                            r = torch.rand(1)
                            if r[0] > 0.5:
                                inputs = torch.zeros_like(inputs)
                            else:
                                local_feature = torch.zeros_like(local_feature)
                        outputs = self.network(inputs, None, local_feature)
                    elif self.sam_network_name == 'sam_with_fullconn_head':
                        feature, _ = self.sam_network(inputs)
                        global_feature, local_feature = feature
                        local_feature = local_feature.detach()
                        global_feature = global_feature.detach()
                        outputs = self.network(inputs, global_feature, local_feature)
                    elif self.sam_network_name == 'sam_with_unet' or self.sam_network_name == 'sam_with_cluster':
                        feature, _ = self.sam_network(inputs)
                        global_feature, local_feature = feature
                        local_feature = local_feature.detach()
                        global_feature = global_feature.detach()
                        outputs = self.network(inputs, global_feature, local_feature)

                else:
                    outputs = self.network(inputs, None, None)

                label_list_batch = batch_data['image_meta_dict']['labels']
                num_of_labels = labels.shape[1]
                batch_size = labels.shape[0]
                loss = torch.tensor(0)
                for i in range(batch_size):
                    label_list = eval(label_list_batch[i])
                    for label_value in range(num_of_labels):
                        if label_value not in label_list:
                            outputs[i,0] = outputs[i,0] + outputs[i,label_value]
                    outputs_t = outputs[i:i+1, label_list]
                    
                    labels_t = labels[i:i+1, label_list]

                    loss_t = self.loss_function(outputs_t, labels_t)
                    loss = loss + loss_t
                loss = loss / batch_size

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            epoch_loss += loss.item()
        self.lr_scheduler.step()
        epoch_loss /= step
        return epoch_loss
    
    def inference(self, input):
        '''
        method to inference data with current model
        '''
        def _foreword(input):
            if self.disabel_sam:
                outputs = self.network(input, None, None)
            else:
                global_feature, local_feature = self.sam_network(input)
                outputs = self.network(input, global_feature, local_feature)
            return outputs

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.roi_size,
                sw_batch_size=1,
                predictor=_foreword,
                overlap=0.5,
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
                    val_data["image"].to(globalVal.device),
                    val_data["label"].to(globalVal.device),
                )
                val_outputs = self.inference(val_inputs)
                val_outputs_bin = torch.where(val_outputs > 0.5, torch.ones_like(val_outputs), torch.zeros_like(val_outputs))
                self.dice_metric(y_pred=val_outputs_bin, y=val_labels)
                self.dice_metric_batch(y_pred=val_outputs_bin, y=val_labels)

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
