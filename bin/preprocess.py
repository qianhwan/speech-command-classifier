from pathlib import Path
from glob import glob
import logging
import argparse


logging.basicConfig()
logger = logging.getLogger('Preprocess')
logger.setLevel(logging.INFO)


def parse_args():
    '''Parse arguments'''
    parser = argparse.ArgumentParser(description="Preprocess")
    parser.add_argument('--data-path', type=str,
                        help='Dataset path')
    parser.add_argument('--output-path', type=str, default='.',
                        help='Path to output metadata csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    Path(args.output_path).mkdir(exist_ok=True)

    subdirs = glob(args.data_path+'/*/')

    if len(subdirs) != 31:
        logger.warning('Expected 31 subfolders, only found %d', len(subdirs))

    validation_list = Path(args.data_path)/'validation_list.txt'
    testing_list = Path(args.data_path)/'testing_list.txt'

    try:
        with validation_list.open() as f:
            val = f.readlines()
        val = [x.strip() for x in val]
        # extend to absolute path
        val = [Path(args.data_path)/x for x in val]
    except FileNotFoundError:
        logger.error('validation_list.txt not found under %s', args.data_path)

    try:
        with testing_list.open() as f:
            test = f.readlines()
        test = [x.strip() for x in test]
        # extend to absolute path
        test = [Path(args.data_path)/x for x in test]
    except FileNotFoundError:
        logger.error('testing_list.txt not found under %s', args.data_path)

    metadata_train = []
    metadata_val = []
    metadata_test = []

    for subdir in subdirs:
        if Path(subdir).name == '_background_noise_':
            continue
        wavs = list(Path(subdir).glob('*'))
        label = Path(subdir).name
        for w in wavs:
            if w in val:
                metadata_val.append('{}\t{}'.format(
                    str(w.absolute()),
                    label))
            elif w in test:
                metadata_test.append('{}\t{}'.format(
                    str(w.absolute()),
                    label))
            else:
                metadata_train.append('{}\t{}'.format(
                    str(w.absolute()),
                    label))

    logger.info('Found %d entries for train', len(metadata_train))
    logger.info('Found %d entries for validation', len(metadata_val))
    logger.info('Found %d entries for testing', len(metadata_test))

    with open(args.output_path + '/metadata_train.csv', 'wt') as f:
        f.write('\n'.join(metadata_train))
    with open(args.output_path + '/metadata_val.csv', 'wt') as f:
        f.write('\n'.join(metadata_val))
    with open(args.output_path + '/metadata_test.csv', 'wt') as f:
        f.write('\n'.join(metadata_test))
