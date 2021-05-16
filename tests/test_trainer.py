import os

import torch

from speech_command_classifier.trainer import Trainer
from speech_command_classifier.data import (SpeechCommandDataset,
                                            collate_fn,
                                            ALL_LABELS)
from speech_command_classifier.model import Model


TRAIN_META = '/home/workspace/metadata/metadata_train.csv'
VAL_META = '/home/workspace/metadata/metadata_val.csv'


train_dataset = SpeechCommandDataset(TRAIN_META,
                                     subset='train',
                                     config={})
val_dataset = SpeechCommandDataset(VAL_META,
                                   subset='validation',
                                   config={})

train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn)

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn)

model = Model(n_input=1, n_output=len(ALL_LABELS))

os.makedirs('/temp/checkpoints', exist_ok=True)
os.makedirs('/temp/models', exist_ok=True)


def test_trainer_build():
    trainer = Trainer({},
                      '/temp',
                      train_dataloader,
                      val_dataloader,
                      model,
                      'cpu')
    assert(trainer)


CHECKPOINT = '/home/workspace/pretrained/checkpoints/best-accuracy-89432.pth'


def test_trainer_load():
    trainer = Trainer({},
                      '/temp',
                      train_dataloader,
                      val_dataloader,
                      model,
                      'cpu')
    trainer.load_checkpoints(CHECKPOINT)
    assert(trainer.steps == 89432)
    assert(trainer.epochs == 55)
    os.removedirs('/temp/checkpoints')
    os.removedirs('/temp/models')
