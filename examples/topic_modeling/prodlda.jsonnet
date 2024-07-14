{
  model: {
    type: 'nlpstack.tasks.topic_modeling.rune.ProdLDA',
    token_indexers: {
      tokens: {
        type: 'nlpstack.data.indexers.SingleIdTokenIndexer',
        lowercase: true,
      },
    },
    min_df: 5,
    max_df: 0.5,
    model: {
      num_topics: 100,
      hidden_dim: 100,
      dropout: 0.2,
    },
    trainer: {
      max_epochs: 100,
      batch_size: 200,
      learning_rate: 0.005,
      callbacks: [
        { type: 'nlpstack.integrations.torch.training.callbacks.EarlyStopping', metric: '+valid_npmi', patience: 10 },
      ],
    },
    metric: [
      { type: 'nlpstack.tasks.topic_modeling.metrics.Perplexity' },
      { type: 'nlpstack.tasks.topic_modeling.metrics.NPMI' },
    ],
  },
  reader: {
    type: 'nlpstack.tasks.topic_modeling.io.TwentyNewsgroupsReader',
  },
  train_dataset_filename: 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz!20news-bydate-train/',
  valid_dataset_filename: 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz!20news-bydate-test/',
}
