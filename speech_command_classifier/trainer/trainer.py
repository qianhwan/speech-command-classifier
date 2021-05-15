'''
Define a trainer object for training
'''
import os
import time
import logging
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_metric

from speech_command_classifier.model import Model


logging.basicConfig()
logger = logging.getLogger('Trainer')
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(
            self,
            config: dict,
            output_dir: str,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            model: Model,
            device: str = 'cpu'):
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 100)
        self.max_training_steps = config.get('max_training_steps', 1000000)
        self.log_interval = config.get('log_interval', 50)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.lr_decay_epochs = config.get('lr_decay_epochs', 20)
        self.lr_decay_gamma = config.get('lr_decay_gamma', 0.1)

        self.output_dir = output_dir
        # create folders for saving checkpoints
        Path(self.output_dir + '/checkpoints').mkdir(exist_ok=True)
        Path(self.output_dir + '/models').mkdir(exist_ok=True)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model

        self.steps = 0
        self.epochs = 0
        self.best_validation_accuracy = 0
        self.best_validation_loss = 100
        self.device = torch.device(device)

        self._set_optimizers()

    def _set_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_decay_epochs,
                gamma=self.lr_decay_gamma)

    def train_epoch(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            loss = F.nll_loss(output.squeeze(), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            log_metric('train_loss', loss.item())

            if batch_idx % self.log_interval == 0:
                logger.info('Epoch: {}, Step: {}, Train Loss: {:.6f}'.format(
                    self.epochs,
                    self.steps,
                    loss.item()))
            self.steps += 1
            self.progress.update(1)

    def validation(self):
        self.model.eval()
        correct = 0
        losses = 0
        for data, target in self.validation_dataloader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)

            losses += F.nll_loss(output.squeeze(), target).item()

            # also compute accuracy
            pred = output.argmax(dim=-1)

            correct += pred.squeeze().eq(target).sum().item()

        accuracy = correct / len(self.validation_dataloader.dataset)
        loss = losses / len(self.validation_dataloader.dataset)
        log_metric('val_loss', loss)
        log_metric('val_acc', accuracy)

        logger.info('Epoch: {}, Validation Loss: {:.6f},'
                    ' Validation Accuracy: {:.0f}%'.format(
                        self.epochs,
                        loss,
                        accuracy*100))

        checkpoint = {
                'epoch': self.epochs,
                'step': self.steps,
                'state_dict': self.model.state_dict(),
                'loss': loss,
                'accuracy': accuracy,
                'optimizer': self.optimizer.state_dict(),
                }

        if accuracy > self.best_validation_accuracy:
            self.best_validation_accuracy = accuracy
            self.save_checkpoints(checkpoint,
                                  'best-accuracy-{}.pth'.format(self.steps))
        if loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.save_checkpoints(checkpoint,
                                  'best-loss-{}.pth'.format(self.steps))

    def train(self):
        self.progress = tqdm(
                initial=self.steps,
                total=self.max_training_steps,
                desc='train')

        init_epochs = self.epochs

        for _ in range(self.num_epochs - init_epochs):
            logger.info('Epoch: %d', self.epochs)
            start_time_epoch = time.time()
            self.train_epoch()
            self.validation()
            self.scheduler.step()
            self.epochs += 1
            end_time_epoch = time.time()
            time_elapsed_epoch = end_time_epoch - start_time_epoch
            logger.info('Time Elapsed for This Epoch: %02d:%02d:%02d',
                        time_elapsed_epoch // 3600,
                        time_elapsed_epoch % 3600 // 60,
                        time_elapsed_epoch % 60 // 1)

    def save_checkpoints(self, checkpoint: dict, name: str):
        torch.save(checkpoint, os.path.join(
            self.output_dir, 'checkpoints', name))
        torch.save(self.model, os.path.join(
            self.output_dir, 'models', name))

    def load_checkpoints(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.float()
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.best_validation_accuracy = checkpoint.get('accuracy', 0)
        self.best_validation_loss = checkpoint.get('loss', 100)
        self.epochs = checkpoint.get('epoch', 0)
        self.steps = checkpoint.get('step', 0)
