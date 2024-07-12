{
  model: {
    type: 'nlpstack.tasks.classification.rune:BasicClassifier',
    max_epochs: 10,
    classifier: {
      type: 'nlpstack.tasks.classification.torch:TorchBasicClassifier',
      embedder: {
        token_embedders: {
          tokens: {
            type: 'nlpstack.integrations.torch.modules.token_embedders:Embedding',
            embedding_dim: 32,
          },
        },
      },
      encoder: {
        type: 'nlpstack.integrations.torch.modules.seq2vec_encoders:BagOfEmbeddings',
        input_dim: 32,
      },
      dropout: 0.1,
    },
  },
  reader: {
    type: 'nlpstack.tasks.classification.io:JsonlReader',
  },
  writer: {
    type: 'nlpstack.tasks.classification.io:JsonlWriter',
  },
  train_dataset_filename: './data/train.jsonl',
  valid_dataset_filename: './data/valid.jsonl',
}
