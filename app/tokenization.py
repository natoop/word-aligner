from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from app.schemas import Token

ChineseTokenize = Callable[[str], Iterable[tuple[str, int, int]]]

_PROTECTED_SOURCE = (
    r"\[\[[^\[\]\r\n]+\]\]"
    r"|\{\{[^{}\r\n]+\}\}"
    r"|\$\{[^{}\r\n]+\}"
    r"|</?[A-Za-z][^>\r\n]*>"
)
_PROTECTED_PATTERN = re.compile(f"(?:{_PROTECTED_SOURCE})")
_CHINESE_LANGUAGE_CODES = {"zh", "yue", "wuu"}
_WORD_JOINERS = {"'", "’", "-"}


def is_protected_text(value: str) -> bool:
    return _PROTECTED_PATTERN.fullmatch(value) is not None


class WordTokenizer:
    """Language-aware word tokenizer that preserves source character offsets."""

    def __init__(self, chinese_tokenize: ChineseTokenize | None = None) -> None:
        self._chinese_tokenize = chinese_tokenize

    def tokenize(self, text: str, language: str) -> list[Token]:
        primary_language = language.split("-", maxsplit=1)[0].lower()
        if primary_language in _CHINESE_LANGUAGE_CODES:
            spans = self._tokenize_chinese(text)
        else:
            spans = self._tokenize_generic(text)

        return [
            Token(
                index=index,
                text=word,
                start=start,
                end=end,
                is_protected=is_protected_text(word),
            )
            for index, (word, start, end) in enumerate(spans)
        ]

    @staticmethod
    def _tokenize_generic(text: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        cursor = 0

        for protected_match in _PROTECTED_PATTERN.finditer(text):
            spans.extend(_tokenize_unicode_fragment(text, cursor, protected_match.start()))
            spans.append((protected_match.group(), protected_match.start(), protected_match.end()))
            cursor = protected_match.end()

        spans.extend(_tokenize_unicode_fragment(text, cursor, len(text)))
        return spans

    def _tokenize_chinese(self, text: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        cursor = 0

        for protected_match in _PROTECTED_PATTERN.finditer(text):
            spans.extend(self._segment_chinese_fragment(text, cursor, protected_match.start()))
            spans.append((protected_match.group(), protected_match.start(), protected_match.end()))
            cursor = protected_match.end()

        spans.extend(self._segment_chinese_fragment(text, cursor, len(text)))
        spans.sort(key=lambda item: (item[1], item[2]))
        return spans

    def _segment_chinese_fragment(self, text: str, start: int, end: int) -> list[tuple[str, int, int]]:
        if start >= end:
            return []

        fragment = text[start:end]
        tokenize = self._chinese_tokenize or self._load_jieba_tokenizer()
        return [
            (word, start + local_start, start + local_end)
            for word, local_start, local_end in tokenize(fragment)
            if word and not word.isspace()
        ]

    def _load_jieba_tokenizer(self) -> ChineseTokenize:
        try:
            import jieba
        except ImportError as exc:  # pragma: no cover - covered by deployment smoke test
            raise RuntimeError("Chinese tokenization requires the 'jieba' package") from exc

        self._chinese_tokenize = lambda value: jieba.tokenize(value, mode="default")
        return self._chinese_tokenize


def _tokenize_unicode_fragment(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    cursor = start

    while cursor < end:
        character = text[cursor]
        if character.isspace():
            cursor += 1
            continue

        if _is_word_start(character):
            token_start = cursor
            cursor += 1
            while cursor < end:
                if _is_word_continuation(text[cursor]):
                    cursor += 1
                    continue
                if (
                    text[cursor] in _WORD_JOINERS
                    and cursor + 1 < end
                    and _is_word_start(text[cursor + 1])
                ):
                    cursor += 1
                    continue
                break
            spans.append((text[token_start:cursor], token_start, cursor))
            continue

        spans.append((character, cursor, cursor + 1))
        cursor += 1

    return spans


def _is_word_start(character: str) -> bool:
    category = unicodedata.category(character)
    return category[0] in {"L", "N"} or category == "Pc"


def _is_word_continuation(character: str) -> bool:
    category = unicodedata.category(character)
    return _is_word_start(character) or category[0] == "M"
