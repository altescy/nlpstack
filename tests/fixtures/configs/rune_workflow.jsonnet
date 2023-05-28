{
  rune: {
    type: 'nlpstack.tasks.classification.rune.BasicClassifier',
    max_epochs: 10,
  },
  reader: {
    type: 'nlpstack.tasks.classification.io.JsonlReader',

  },
  writer: {
    type: 'nlpstack.tasks.classification.io.JsonlWriter',
  },
  train_dataset_filename: './tests/fixtures/data/classification.jsonl',
  valid_dataset_filename: './tests/fixtures/data/classification.jsonl',
}
