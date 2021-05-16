# speech-command-classifier


## Build docker

```sh
docker built -t speech-command .
docker run -it -p 8888:8888 \
    -v /home/qianhui/Dataset/speech_commands_v0.01/:/data
    -v $PWD:/home/workspace
    speech-command /bin/bash 
```

## Preprocess dataset

```sh
python bin/preprocess.py \
    --data-path /data/ \
    --output-path metadata/
```

This will create three metadata csv files (for train, validation and test respectively) inside OUT-DIR, each csv file has two columns: audio file path and label.


## Train

Use the following command to train model, make sure you run preprocess.py first.

```sh
python bin/train.py \
    --train-metadata metadata/metadata_train.csv \
    --validation-metadata metadata/metadata_val.csv \
    --config configs/config.yaml \
    --output outputs/
```

Use `--resume` to continue training from a previously saved checkpoint.

### Monitor training and validation loss

```sh
mlflow ui
```

## Predict

```sh
python bin/predict.py \
    --audio /data/cat/0819edb0_nohash_0.wav \
    --model models/best-accuracy-229896.pth \
    --config configs/config.yaml
```
