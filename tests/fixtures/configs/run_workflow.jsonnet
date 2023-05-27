{
  rune: {
    type: 'nlpstack.tasks.classification.rune.BasicClassifier',
    max_epochs: 10,
  },
  reader: {
    type: 'nlpstack.tasks.classification.io.JsonlReader',
    train_filename: './tests/fixtures/data/classification.jsonl',
    valid_filename: './tests/fixtures/data/classification.jsonl',
  },
  writer: {
    type: 'nlpstack.tasks.classification.io.JsonlWriter',
  },
}
