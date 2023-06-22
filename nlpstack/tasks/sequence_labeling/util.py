from typing import List, Optional, Sequence, Set, Tuple

LabelWithSpan = Tuple[str, Tuple[int, int]]


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self) -> str:
        if self.tag_sequence is None:
            return "Invalid tag sequence"
        return " ".join(self.tag_sequence)


def bio_tags_to_spans(
    tag_sequence: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag: Optional[str] = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ("B", "I", "O"):
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def iob1_tags_to_spans(
    tag_sequence: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    def _iob1_start_of_chunk(
        prev_bio_tag: Optional[str],
        prev_conll_tag: Optional[str],
        curr_bio_tag: str,
        curr_conll_tag: str,
    ) -> bool:
        if curr_bio_tag == "B":
            return True
        if curr_bio_tag == "I" and prev_bio_tag == "O":
            return True
        if curr_bio_tag != "O" and prev_conll_tag != curr_conll_tag:
            return True
        return False

    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag: Optional[str] = None
    prev_bio_tag: Optional[str] = None
    prev_conll_tag: Optional[str] = None
    for index, string_tag in enumerate(tag_sequence):
        curr_bio_tag = string_tag[0]
        curr_conll_tag = string_tag[2:]

        if curr_bio_tag not in ("B", "I", "O"):
            raise InvalidTagSequence(tag_sequence)
        if curr_bio_tag == "O" or curr_conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            # bio_tag == "I" and curr_conll_tag == active_conll_tag
            # We're continuing a span.
            span_end += 1

        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def bioul_tags_to_spans(
    tag_sequence: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            spans.append((label.partition("-")[2], (index, index)))
        elif label[0] == "B":
            start = index
            while label[0] != "L":
                index += 1
                if index >= len(tag_sequence):
                    raise InvalidTagSequence(tag_sequence)
                label = tag_sequence[index]
                if not label[0] in ("I", "L"):
                    raise InvalidTagSequence(tag_sequence)
            spans.append((label.partition("-")[2], (start, index)))
        else:
            if label != "O":
                raise InvalidTagSequence(tag_sequence)
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def bmes_tags_to_spans(
    tag_sequence: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    def extract_bmes_tag_label(text: str) -> Tuple[str, str]:
        bmes_tag = text[0]
        label = text[2:]
        return bmes_tag, label

    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for index, tag in enumerate(tag_sequence):
        bmes_tag, label = extract_bmes_tag_label(tag)
        if bmes_tag in ("B", "S"):
            # Regardless of tag, we start a new span when reaching B & S.
            spans.append((label, [index, index]))
        elif bmes_tag in ("M", "E") and prev_bmes_tag in ("B", "M") and spans[-1][0] == label:
            # Only expand the span if
            # 1. Valid transition: B/M -> M/E.
            # 2. Matched label.
            spans[-1][1][1] = index
        else:
            # Best effort split for invalid span.
            spans.append((label, [index, index]))
        # update previous BMES tag.
        prev_bmes_tag = bmes_tag

    classes_to_ignore = classes_to_ignore or []
    return [
        # to tuple.
        (span[0], (span[1][0], span[1][1]))
        for span in spans
        if span[0] not in classes_to_ignore
    ]
