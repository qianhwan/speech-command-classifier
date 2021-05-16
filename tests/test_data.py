from speech_command_classifier.data import SpeechCommandDataset


# this is fixed because it's our working directory inside docker
METADATA = '/home/workspace/metadata/metadata_train.csv'


def test_dataset_length():
    dataset = SpeechCommandDataset(METADATA, {})
    assert(len(dataset) == 51088)


def test_dataset_feats():
    dataset = SpeechCommandDataset(METADATA, {})
    item = next(iter(dataset))
    assert(item['feats'].shape[-1] == 32)
    assert(isinstance(item['label'], int))

