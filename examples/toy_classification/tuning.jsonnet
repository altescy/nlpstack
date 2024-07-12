local embedding_dim = std.parseJson(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local learning_rate = std.parseJson(std.extVar('learning_rate'));

{
  model: {
    type: 'nlpstack.tasks.classification.rune.BasicClassifier',
    max_epochs: 10,
    learning_rate: learning_rate,
    classifier: {
      type: 'nlpstack.tasks.classification.torch.TorchBasicClassifier',
      embedder: {
        token_embedders: {
          tokens: {
            type: 'nlpstack.integrations.torch.modules.token_embedders.Embedding',
            embedding_dim: embedding_dim,
          },
        },
      },
      encoder: {
        type: 'nlpstack.integrations.torch.modules.seq2vec_encoders.BagOfEmbeddings',
        input_dim: embedding_dim,
      },
      dropout: dropout,
    },
    training_callbacks: [
      {
        type: 'nlpstack.integrations.torch.training.callbacks.MlflowCallback',
      },
    ],
  },
  reader: {
    type: 'nlpstack.tasks.classification.io.JsonlReader',
  },
  writer: {
    type: 'nlpstack.tasks.classification.io.JsonlWriter',
  },
  train_dataset_filename: './data/train.jsonl',
  valid_dataset_filename: './data/valid.jsonl',
}
