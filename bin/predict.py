'''
Given an input audio, make a prediction
'''
import yaml
import logging
import argparse

import torch

from speech_command_classifier.data import (
        ALL_LABELS,
        load_audio,
        normalize_signal,
        normalize_audio_feature,
        compute_log_mel_spectrogram)


logging.basicConfig()
logger = logging.getLogger('predict')
logger.setLevel(logging.INFO)


def parse_args():
    '''Parse arguments'''
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument('--audio', type=str,
                        help='input audio path')
    parser.add_argument('--model', type=str,
                        help='pretrained model path')
    parser.add_argument('--config', type=str,
                        help='configuration file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    signal = load_audio(args.audio)

    normalize_audio = config['preprocess_params'].get('normalize_audio', True)
    normalize_feature = config['preprocess_params'].get('normalize_feature',
                                                        True)
    num_feature_bins = config['preprocess_params'].get('num_feature_bins',
                                                        32)

    if normalize_audio:
        signal = normalize_signal(signal)

    feats = compute_log_mel_spectrogram(signal, num_feature_bins)

    if normalize_feature:
        feats = normalize_audio_feature(feats)
    logger.info('Extracted features from audio')

    model = torch.load(args.model)
    model.float()
    logger.info('Loaded model weights')

    output = model(torch.tensor(feats).unsqueeze(0))
    pred = output.argmax(dim=-1).item()
    logger.info('Predict command: %s', ALL_LABELS[pred])
