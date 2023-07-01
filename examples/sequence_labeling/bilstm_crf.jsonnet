local label_encoding = 'BIOUL';
local token_character_namespace = 'token_characters';

{
  model: {
    type: 'nlpstack.tasks.sequence_labeling.rune.BasicSequenceLabeler',
    token_indexers: {
      token_characters: {
        type: 'nlpstack.data.indexers.TokenCharactersIndexer',
        namespace: token_character_namespace,
        min_padding_length: 4,
      },
    },
    sequence_labeler: {
      type: 'nlpstack.tasks.sequence_labeling.torch.TorchSequenceLabeler',
      embedder: {
        token_embedders: {
          token_characters: {
            type: 'nlpstack.torch.modules.token_embedders.TokenSubwordsEmbedder',
            embedder: {
              type: 'nlpstack.torch.modules.token_embedders.Embedding',
              embedding_dim: 64,
              namespace: token_character_namespace,
            },
            encoder: {
              type: 'nlpstack.torch.modules.seq2vec_encoders.CnnEncoder',
              input_dim: 64,
              num_filters: 16,
              ngram_filter_sizes: [1, 2, 3, 4],
            },
            dropout: 0.1,
          },
        },
      },
      encoder: {
        type: 'nlpstack.torch.modules.seq2seq_encoders.LstmSeq2SeqEncoder',
        input_dim: 64,
        hidden_dim: 32,
        num_layers: 1,
        bidirectional: true,
      },
      decoder: {
        constraint: label_encoding,
      },
      dropout: 0.5,
    },
    metric: [
      {
        type: 'nlpstack.tasks.sequence_labeling.metrics.SpanBasedF1',
        label_encoding: label_encoding,
      },
    ],
    trainer: {
      max_epochs: 80,
      batch_size: 32,
      optimizer_factory: {
        type: 'nlpstack.torch.training.optimizers.AdamFactory',
        lr: 0.01,
      },
      callbacks: [
        {
          type: 'nlpstack.torch.training.callbacks.EarlyStopping',
          patience: 3,
          metric: '+valid_f1_overall',
        },
      ],
    },
  },
  reader: {
    type: 'nlpstack.tasks.sequence_labeling.io.Conll2003Reader',
    label_field: 'ner',
    convert_to_label_encoding: label_encoding,
  },
  train_dataset_filename: std.extVar('TRAIN_DATASET_FILENAME'),
  valid_dataset_filename: std.extVar('VALID_DATASET_FILENAME'),
}
