import itertools
from typing import List, Literal, Optional, Sequence, Tuple

LabelWithSpan = Tuple[str, Tuple[int, int]]
LabelEncoding = Literal["BIO", "IOB1", "BIOUL", "BMES"]


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self) -> str:
        if self.tag_sequence is None:
            return "Invalid tag sequence"
        return " ".join(self.tag_sequence)


def bioul_tags_to_spans(
    tags: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    spans = []
    classes_to_ignore = classes_to_ignore or []
    for index, tag in enumerate(tags):
        if tag == "O":
            continue

        split = tag.split("-", 1)
        if len(split) != 2:
            raise InvalidTagSequence(tags)

        prefix, label = split
        if label in classes_to_ignore:
            continue

        if prefix in ("B", "U"):
            spans.append((label, (index, index)))
        elif prefix in ("I", "L"):
            if not spans or spans[-1][0] != label:
                raise InvalidTagSequence(tags)
            spans[-1] = (label, (spans[-1][1][0], index))
        else:
            raise InvalidTagSequence(tags)
    return spans


def bio_tags_to_spans(
    tags: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    return bioul_tags_to_spans(bio_to_bioul(tags), classes_to_ignore)


def bmes_tags_to_spans(
    tags: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    return bioul_tags_to_spans(bmes_to_bioul(tags), classes_to_ignore)


def iob1_tags_to_spans(
    tags: Sequence[str],
    classes_to_ignore: Optional[Sequence[str]] = None,
) -> List[LabelWithSpan]:
    return bioul_tags_to_spans(iob1_to_bioul(tags), classes_to_ignore)


def spans_to_bioul_tags(spans: Sequence[LabelWithSpan], length: int) -> List[str]:
    tags = ["O"] * length
    for label, (start, end) in spans:
        if start == end:
            tags[start] = f"U-{label}"
        else:
            tags[start] = f"B-{label}"
            tags[end] = f"L-{label}"
            for i in range(start + 1, end):
                tags[i] = f"I-{label}"
    return tags


def spans_to_bio_tags(spans: Sequence[LabelWithSpan], length: int) -> List[str]:
    return bioul_to_bio(spans_to_bioul_tags(spans, length))


def spans_to_bmes_tags(spans: Sequence[LabelWithSpan], length: int) -> List[str]:
    return bioul_to_bmes(spans_to_bioul_tags(spans, length))


def spans_to_iob1_tags(spans: Sequence[LabelWithSpan], length: int) -> List[str]:
    return bioul_to_iob1(spans_to_bioul_tags(spans, length))


def iob1_to_bioul(tags: Sequence[str]) -> List[str]:
    def convert(preceding: str, current: str, following: str) -> str:
        if current == "O":
            return "O"

        if current.startswith("B-"):
            if following == f"I-{current[2:]}":
                return current
            if following == "O" or following.startswith("B-") or following != f"I-{current[2:]}":
                return f"U-{current[2:]}"
            return current

        if current.startswith("I-"):
            if preceding != current and following == current:
                return f"B-{current[2:]}"
            if following == "O" or following.startswith("B-") or following != f"I-{current[2:]}":
                if preceding in (current, f"B-{current[2:]}"):
                    return f"L-{current[2:]}"
                return f"U-{current[2:]}"
            return f"I-{current[2:]}"

        raise InvalidTagSequence(tags)

    return list(
        itertools.starmap(
            convert,
            zip(
                itertools.chain("O", tags),
                tags,
                itertools.chain(tags[1:], "O"),
            ),
        )
    )


def bio_to_bioul(tags: Sequence[str]) -> List[str]:
    def convert(preceding: str, current: str, following: str) -> str:
        if current == "O":
            return "O"

        if current.startswith("B-"):
            if following == f"I-{current[2:]}":
                return current
            if following == "O" or following.startswith("B-") or following != f"I-{current[2:]}":
                return f"U-{current[2:]}"
            return current

        if current.startswith("I-"):
            if following == "O" or following.startswith("B-") or following != f"I-{current[2:]}":
                return f"L-{current[2:]}"
            return f"I-{current[2:]}"

        raise InvalidTagSequence(tags)

    return list(
        itertools.starmap(
            convert,
            zip(
                itertools.chain("O", tags),
                tags,
                itertools.chain(tags[1:], "O"),
            ),
        )
    )


def bmes_to_bioul(tags: Sequence[str]) -> List[str]:
    return [tag.replace("M-", "I-").replace("E-", "L-") for tag in tags]


def bioul_to_iob1(tags: Sequence[str]) -> List[str]:
    def convert(preceding: str, current: str, following: str) -> str:
        if current == "O":
            return "O"

        if current.startswith(("B-", "U-")):
            if preceding != "O" and preceding[2:] == current[2:]:
                return f"B-{current[2:]}"
            return f"I-{current[2:]}"

        if current.startswith(("I-", "L-")):
            return f"I-{current[2:]}"

        raise InvalidTagSequence(tags)

    return list(
        itertools.starmap(
            convert,
            zip(
                itertools.chain("O", tags),
                tags,
                itertools.chain(tags[1:], "O"),
            ),
        )
    )


def bioul_to_bio(tags: Sequence[str]) -> List[str]:
    return [tag.replace("L-", "I-").replace("U-", "B-") for tag in tags]


def bioul_to_bmes(tags: Sequence[str]) -> List[str]:
    return [tag.replace("I-", "M-").replace("L-", "E-") for tag in tags]


def bio_to_iob1(tags: Sequence[str]) -> List[str]:
    return bioul_to_iob1(bio_to_bioul(tags))


def iob1_to_bio(tags: Sequence[str]) -> List[str]:
    return bioul_to_bio(iob1_to_bioul(tags))


def bio_to_bmes(tags: Sequence[str]) -> List[str]:
    return bioul_to_bmes(bio_to_bioul(tags))


def bmes_to_bio(tags: Sequence[str]) -> List[str]:
    return bioul_to_bio(bmes_to_bioul(tags))


def iob1_to_bmes(tags: Sequence[str]) -> List[str]:
    return bioul_to_bmes(iob1_to_bioul(tags))


def bmes_to_iob1(tags: Sequence[str]) -> List[str]:
    return bioul_to_iob1(bmes_to_bioul(tags))


def tags_to_spans(tags: Sequence[str], encoding: LabelEncoding) -> List[LabelWithSpan]:
    if encoding == "BIO":
        return bio_tags_to_spans(tags)
    if encoding == "BIOUL":
        return bioul_tags_to_spans(tags)
    if encoding == "BMES":
        return bmes_tags_to_spans(tags)
    if encoding == "IOB1":
        return iob1_tags_to_spans(tags)
    raise ValueError(f"Invalid encoding: {encoding}")


def spans_to_tags(spans: Sequence[LabelWithSpan], length: int, encoding: LabelEncoding) -> List[str]:
    if encoding == "BIO":
        return spans_to_bio_tags(spans, length)
    if encoding == "BIOUL":
        return spans_to_bioul_tags(spans, length)
    if encoding == "BMES":
        return spans_to_bmes_tags(spans, length)
    if encoding == "IOB1":
        return spans_to_iob1_tags(spans, length)
    raise ValueError(f"Invalid encoding: {encoding}")


def convert_encoding(
    tags: Sequence[str],
    source_encoding: LabelEncoding,
    target_encoding: LabelEncoding,
) -> List[str]:
    if source_encoding == target_encoding:
        return list(tags)
    if (source_encoding, target_encoding) == ("BIO", "BIOUL"):
        return bio_to_bioul(tags)
    if (source_encoding, target_encoding) == ("BIO", "BMES"):
        return bio_to_bmes(tags)
    if (source_encoding, target_encoding) == ("BIO", "IOB1"):
        return bio_to_iob1(tags)
    if (source_encoding, target_encoding) == ("BIOUL", "BIO"):
        return bioul_to_bio(tags)
    if (source_encoding, target_encoding) == ("BIOUL", "BMES"):
        return bioul_to_bmes(tags)
    if (source_encoding, target_encoding) == ("BIOUL", "IOB1"):
        return bioul_to_iob1(tags)
    if (source_encoding, target_encoding) == ("BMES", "BIO"):
        return bmes_to_bio(tags)
    if (source_encoding, target_encoding) == ("BMES", "BIOUL"):
        return bmes_to_bioul(tags)
    if (source_encoding, target_encoding) == ("BMES", "IOB1"):
        return bmes_to_iob1(tags)
    if (source_encoding, target_encoding) == ("IOB1", "BIO"):
        return iob1_to_bio(tags)
    if (source_encoding, target_encoding) == ("IOB1", "BIOUL"):
        return iob1_to_bioul(tags)
    if (source_encoding, target_encoding) == ("IOB1", "BMES"):
        return iob1_to_bmes(tags)
    raise ValueError(f"Invalid encoding pair: {source_encoding}, {target_encoding}")
