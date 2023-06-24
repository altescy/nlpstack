local label_encoding = 'BIOUL';

{
  model: {
    type: 'nlpstack.tasks.sequence_labeling.rune.BasicSequenceLabeler',
    sequence_labeler: {
      type: 'nlpstack.tasks.sequence_labeling.torch.TorchSequenceLabeler',
      embedder: {
        token_embedders: {
          tokens: {
            type: 'nlpstack.torch.modules.token_embedders.Embedding',
            embedding_dim: 64,
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
      dropout: 0.3,
    },
    metric: [
      {
        type: 'nlpstack.tasks.sequence_labeling.metrics.SpanBasedF1',
        label_encoding: label_encoding,
      },
    ],
    trainer: {
      max_epochs: 10,
      batch_size: 32,
      optimizer_factory: {
        type: 'nlpstack.torch.training.optimizers.AdamFactory',
        lr: 0.001,
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
