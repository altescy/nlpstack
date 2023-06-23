# Sequence Labeling Example

```python
export TRAIN_DATASET_FILENAME=path/to/conll2003/train.txt
export VALID_DATASET_FILENAME=path/to/conll2003/valid.txt

# Training
nlpstack workflow rune train config.jsonnet archive.pkl

# Serve model
nlpstack serve archive.pkl
```
