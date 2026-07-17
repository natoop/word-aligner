from __future__ import annotations

import re

from app.tokenization import WordTokenizer


def _dictionary_chinese_tokenize(value: str):
    words = ("我", "喜欢", "机器翻译", "。")
    cursor = 0
    while cursor < len(value):
        if value[cursor].isspace():
            cursor += 1
            continue
        word = next((candidate for candidate in words if value.startswith(candidate, cursor)), value[cursor])
        yield word, cursor, cursor + len(word)
        cursor += len(word)


def _whitespace_tokenize(value: str):
    for match in re.finditer(r"\S+", value):
        yield match.group(), match.start(), match.end()


def test_generic_tokenizer_preserves_offsets_and_protected_tokens() -> None:
    text = "Keep [[T1504_1]] and ${name} intact."

    tokens = WordTokenizer().tokenize(text, "en")

    assert [token.text for token in tokens] == ["Keep", "[[T1504_1]]", "and", "${name}", "intact", "."]
    assert [text[token.start : token.end] for token in tokens] == [token.text for token in tokens]
    assert [token.text for token in tokens if token.is_protected] == ["[[T1504_1]]", "${name}"]


def test_chinese_tokenizer_returns_word_level_tokens_with_offsets() -> None:
    text = "我喜欢机器翻译。"
    tokenizer = WordTokenizer(chinese_tokenize=_dictionary_chinese_tokenize)

    tokens = tokenizer.tokenize(text, "zh-Hans")

    assert [token.text for token in tokens] == ["我", "喜欢", "机器翻译", "。"]
    assert [(token.start, token.end) for token in tokens] == [(0, 1), (1, 3), (3, 7), (7, 8)]


def test_chinese_tokenizer_does_not_split_a_placeholder() -> None:
    text = "保留 [[T1504_1]] 不变"
    tokenizer = WordTokenizer(chinese_tokenize=_whitespace_tokenize)

    tokens = tokenizer.tokenize(text, "zh-Hans")

    placeholder = next(token for token in tokens if token.text == "[[T1504_1]]")
    assert placeholder.is_protected is True
    assert text[placeholder.start : placeholder.end] == "[[T1504_1]]"


def test_tokenizer_preserves_index_arrow_expressions_as_protected_tokens() -> None:
    text = "保留 5→3 和 6->4"
    tokenizer = WordTokenizer(chinese_tokenize=_whitespace_tokenize)

    tokens = tokenizer.tokenize(text, "zh-Hans")

    assert [token.text for token in tokens] == ["保留", "5→3", "和", "6->4"]
    assert [token.text for token in tokens if token.is_protected] == ["5→3", "6->4"]
    assert [text[token.start : token.end] for token in tokens] == [token.text for token in tokens]


def test_unicode_tokenizer_keeps_combining_marks_inside_indic_words() -> None:
    text = "मैं हिन्दी बोलता हूँ।"

    tokens = WordTokenizer().tokenize(text, "hi")

    assert [token.text for token in tokens] == ["मैं", "हिन्दी", "बोलता", "हूँ", "।"]
    assert [text[token.start : token.end] for token in tokens] == [token.text for token in tokens]
