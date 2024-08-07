local label_encoding = 'BIOUL';
local pretrained_model_name = 'bert-base-cased';

{
  model: {
    type: 'nlpstack.tasks.sequence_labeling.rune.BasicSequenceLabeler',
    pad_token: '[PAD]',
    oov_token: '[UNK]',
    token_indexers: {
      tokens: {
        type: 'nlpstack.data.indexers.PretrainedTransformerIndexer',
        pretrained_model_name: pretrained_model_name,
        tokenize_subwords: true,
      },
    },
    sequence_labeler: {
      type: 'nlpstack.tasks.sequence_labeling.torch.TorchSequenceLabeler',
      embedder: {
        token_embedders: {
          tokens: {
            type: 'nlpstack.integrations.torch.modules.token_embedders.AggregativeTokenEmbedder',
            embedder: {
              type: 'nlpstack.integrations.torch.modules.token_embedders.PretrainedTransformerEmbedder',
              pretrained_model_name: pretrained_model_name,
              train_parameters: false,
              layer_to_use: 'all',
            },
            dropout: 0.1,
          },
        },
      },
      encoder: {
        type: 'nlpstack.integrations.torch.modules.seq2seq_encoders.PassThroughSeq2SeqEncoder',
        input_dim: 768,
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
      max_epochs: 80,
      batch_size: 32,
      optimizer_factory: {
        type: 'nlpstack.integrations.torch.training.optimizers.AdamFactory',
        lr: 0.01,
      },
      callbacks: [
        {
          type: 'nlpstack.integrations.torch.training.callbacks.EarlyStopping',
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
