{
  model: {
    type: 'nlpstack.tasks.keyphrase_extraction:CValue',
    tokenizer: {
      type: 'nlpstack.data:SpacyTokenizer',
      lang: 'en_core_web_sm',
      with_whitespace: true,
    },
    candidate_postag_pattern: '^((<[A-Z]+:[A-Z]+><[A-Z]+:HYPH><[A-Z]+:[A-Z]+>)(<NOUN:[A-Z]+>)*)|((<VERB:(VBG|VBN)>)?(<ADJ:[A-Z]+>)*(<NOUN:[A-Z]+>)+)$',
    metric: {
      type: 'nlpstack.tasks.keyphrase_extraction.metrics:FBeta',
      topk: 10,
      ignore_case: true,
    },
    nc_weight: 0.2,
  },
  reader: { type: 'inspec.InspecDatasetReader' },
  writer: { type: 'inspec.JsonlWriter' },
  train_dataset_filename: 'https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zip!Inspec',
  valid_dataset_filename: 'https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zip!Inspec',
}
