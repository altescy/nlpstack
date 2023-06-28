# Sequence Labeling Example

```shell
export TRAIN_DATASET_FILENAME=path/to/conll2003/train.txt
export VALID_DATASET_FILENAME=path/to/conll2003/valid.txt

# Training
nlpstack workflow rune train config.jsonnet archive.tar.gz

# Serve model
nlpstack serve archive.tar.gz
```
