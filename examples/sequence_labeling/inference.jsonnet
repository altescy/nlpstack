{
  reader: {
    type: 'nlpstack.tasks.sequence_labeling.io.Conll2003Reader',
    label_field: 'ner',
    convert_to_label_encoding: 'BIOUL',
  },
  writer: {
    type: 'nlpstack.tasks.sequence_labeling.io.JsonlWriter',
  },
}
