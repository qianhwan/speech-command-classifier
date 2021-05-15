import yaml
import argparse
from pathlib import Path

import torch

from speech_command_classifier.data import (
        SpeechCommandDataset, collate_fn, ALL_LABELS)
from speech_command_classifier.model import Model
from speech_command_classifier.trainer import Trainer


def parse_args():
    '''Parse arguments'''
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--train-metadata', type=str,
                        help='train_metadata.csv')
    parser.add_argument('--validation-metadata', type=str,
                        help='validation_metadata.csv')
    parser.add_argument('--config', type=str,
                        help='configuration file')
    parser.add_argument('--output', type=str,
                        help='output directory')
    parser.add_argument('--resume', type=str, default='',
                        help='checkpoint to resume training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    Path(args.output).mkdir(exist_ok=True)

    train_dataset = SpeechCommandDataset(args.train_metadata, subset='train',
                                         config=config['preprocess_params'])
    val_dataset = SpeechCommandDataset(args.validation_metadata,
                                       subset='validation',
                                       config=config['preprocess_params'])

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['trainer_params']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['trainer_params']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn)

    model = Model(n_input=1, n_output=len(ALL_LABELS))

    if args.resume:
        model.load_checkpoints(args.resume)

    trainer = Trainer(config['trainer_params'],
                      args.output,
                      train_dataloader,
                      val_dataloader,
                      model,
                      'cpu')
    trainer.train()
