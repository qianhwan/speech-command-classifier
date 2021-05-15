import logging

import torch
from torch.utils.data.dataset import Dataset

from speech_command_classifier.data.audio import (
        load_audio,
        fix_audio_length,
        normalize_signal,
        normalize_audio_feature,
        compute_log_mel_spectrogram)
from speech_command_classifier.data.augment import (
        ChangeAmplitude,
        ChangeSpeedAndPitchAudio,
        TimeshiftAudio,
        StretchAudio)


logging.basicConfig()
logger = logging.getLogger('SpeechCommandDataset')
logger.setLevel(logging.INFO)


# copied from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html#define-the-network  # noqa
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # feats, label

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for item in batch:
        tensors += [torch.tensor(item['feats'])]
        targets += [torch.tensor(item['label'])]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


ALL_LABELS = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four',
              'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off',
              'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
              'tree', 'two', 'up', 'wow', 'yes', 'zero']


class SpeechCommandDataset(Dataset):
    '''
    Define speech command dataset for data loading
    '''
    def __init__(self,
                 metadata: str,
                 config: dict,
                 subset: str = 'train',
                 labels=ALL_LABELS):
        '''
        :param str metadata: metadata csv file that has three columns
                             audio file path, label
        :param str subset: train, validation or test
        :param dict config: configurations for feature extraction
        '''
        with open(metadata) as f:
            metadata = f.readlines()
        self.data = [x.strip().split('\t') for x in metadata]

        self.subset = subset

        self.normalize_audio = config.get('normalize_audio', True)
        self.normalize_feature = config.get('normalize_feature', True)
        self.sample_rate = config.get('sample_rate', 16000)
        self.noise_augmentation = config.get('noise_augmentation', False)
        self.num_feature_bins = config.get('num_feature_bins', 32)
        self.audio_length = config.get('audio_length', 1)

        # augmentation
        self.augment_amplitude = config.get('augment_amplitude', False)
        self.augment_speed = config.get('augment_speed', False)
        self.augment_stretch = config.get('augment_stretch', False)
        self.augment_shift = config.get('augment_shift', False)

        # get all labels
        self.all_labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        signal = load_audio(self.data[index][0])

        if self.normalize_audio:
            signal = normalize_signal(signal)

        if self.augment_amplitude:
            signal = ChangeAmplitude()(signal)
        if self.augment_speed:
            signal = ChangeSpeedAndPitchAudio()(signal)
        if self.augment_stretch:
            signal = StretchAudio()(signal)
        if self.augment_shift:
            signal = TimeshiftAudio()(signal, self.sample_rate)
        signal = fix_audio_length(signal, self.sample_rate * self.audio_length)

        feats = compute_log_mel_spectrogram(signal, self.num_feature_bins)

        if self.normalize_feature:
            feats = normalize_audio_feature(feats)

        # convert label to id
        label = self.data[index][1]
        label = self.all_labels.index(label)

        sample = {'feats': feats, 'label': label}
        return sample
