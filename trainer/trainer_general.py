import time
import torch
import pickle
from monai.metrics import DiceMetric

from trainer.base_trainer import base_trainer
from utility import *

class trainer_general(base_trainer):
    '''
    general trainer
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        super(trainer_general, self).__init__(config)

        self.max_epochs = int(config['Train']['max_epochs'])
        self.save_every_epochs = int(config['Train']['save_every_epochs'])
        self.val_interval = int(config['Train']['val_interval'])
        self.continue_training = bool(config['Train']['continue_training'])
        self.val_amp = bool(config['Train']['val_amp'])

        self.epoch = 0
        self.start_epoch = 0
        self.metric_log = {'metric_list': [], 'best_metric_list': []}

        if self.continue_training:
            self.load_latest_checkpoint()
        
        self.generate_dataloader()
    
    def generate_dataloader(self):
        self.train_loader = self.dataloader.get_train_loader()
        self.val_loader = self.dataloader.get_val_loader()

    def run_training(self):
        # use amp to accelerate training
        scaler = torch.cuda.amp.GradScaler()
        # enable cuDNN benchmark
        torch.backends.cudnn.benchmark = True

        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            training_metric_dict = {'epoch':self.epoch + 1}

            epoch_start = time()
            print_to_log_file("-" * 20)
            print_to_log_file(f"epoch {epoch + 1}/{self.max_epochs}")
            
            epoch_loss = self.train_epoch(scaler)
            if self.epoch == 0:
                self.best_epoch_loss = epoch_loss

            training_metric_dict['train_loss'] = epoch_loss
            print_to_log_file(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (self.epoch+1) % self.save_every_epochs == 0:
                self.save_checkpoint("latest_metric_model.pth")
                print_to_log_file("saved new latest metric checkpoint")

            if epoch_loss < self.best_epoch_loss:
                self.best_epoch_loss = epoch_loss
                self.save_checkpoint("best_train_loss_model.pth")
                print_to_log_file("saved new best metric checkpoint")

            if self.val_interval > 0 and (epoch + 1) % self.val_interval == 0:
                self.validation_eval(training_metric_dict)

            print_to_log_file(f"time consuming of epoch {self.epoch + 1} is: {(time() - epoch_start):.4f}")

            self.metric_log['metric_list'].append(training_metric_dict)
            self.save_metric()

    def validation_eval(self, training_metric_dict):
        pass
    
    def save_checkpoint(self, name):
        if self.ddp_mode and not globalVal.first_rank:
            return
        if self.ddp_mode:
            checkpoint = {
                "net": self.network.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_evaluation": self.best_epoch_loss,
                "metric_log": self.metric_log,
            }
        else:
            checkpoint = {
                "net": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_evaluation": self.best_epoch_loss,
                "metric_log": self.metric_log,
            }
        torch.save(checkpoint, join(globalVal.model_path, name))

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
    
    def save_metric(self):
        if self.ddp_mode and not globalVal.first_rank:
            return
        matric_name = join(globalVal.model_path, 'training_matrics.pkl')
        if isfile(matric_name):
            os.remove(matric_name)
        f_save = open(matric_name, 'wb')
        pickle.dump(self.metric_log, f_save)
        f_save.close()
        print_to_log_file("saved matrics log file")

    def load_latest_checkpoint(self):
        checkpoint_path = join(globalVal.model_path, "latest_metric_model.pth")
        assert isfile(checkpoint_path), "error: checkpoint file does not exist. from trainer_general.py"
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_epoch_loss = checkpoint['best_evaluation']
        self.metric_log = checkpoint['metric_log']
