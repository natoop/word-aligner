from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

LanguageCode = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=35,
        pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$",
    ),
]
NonEmptyText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=10_000),
]
AlignmentMethod = Literal["itermax", "inter", "mwmf"]
AlignmentType = Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
AlignmentLinkOrigin = Literal["model", "rule", "repaired"]
AlignmentGroupOrigin = Literal["model", "rule", "repaired", "refined", "mixed"]
ConfidenceMethod = Literal["bidirectional-margin-span-v2"]
WordTokenizerType = Literal["unicode-regex", "jieba"]


class SentencePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=100)
    source: NonEmptyText
    target: NonEmptyText

    @field_validator("source", "target")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain at least one non-whitespace character")
        return value


class RepairOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    strategy: Literal["conservative", "span-aware"] = "conservative"
    max_position_distance: float = Field(default=0.35, ge=0.0, le=1.0)
    min_similarity: float = Field(default=0.45, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    max_source_span: int = Field(
        default=3,
        ge=1,
        le=32,
        description="Maximum source-token count per local refined span",
    )
    max_target_span: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Maximum target-token count per local refined span",
    )
    min_score_gain: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum score gain for span-aware and composite repair-island expansion",
    )
    min_span_coverage: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Coverage threshold for conservative clause refinement",
    )


class AlignmentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "source_language": "en",
                "target_language": "zh-Hans",
                "method": "itermax",
                "repair": {
                    "enabled": True,
                    "strategy": "conservative",
                    "max_position_distance": 0.35,
                    "min_similarity": 0.45,
                    "min_confidence": 0.35,
                    "max_source_span": 3,
                    "max_target_span": 6,
                    "min_score_gain": 0.05,
                    "min_span_coverage": 0.75,
                },
                "sentence_pairs": [
                    {
                        "id": "sentence-1",
                        "source": "This is a machine translation example.",
                        "target": "这是一个机器翻译示例。",
                    }
                ],
            }
        },
    )

    source_language: LanguageCode
    target_language: LanguageCode
    sentence_pairs: list[SentencePair] = Field(min_length=1, max_length=100)
    method: AlignmentMethod = "itermax"
    repair: RepairOptions | None = None


class Token(BaseModel):
    index: int
    text: str
    start: int
    end: int
    is_protected: bool = False


class AlignmentLink(BaseModel):
    source_index: int
    target_index: int
    origin: AlignmentLinkOrigin = "model"
    similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class AlignmentGroup(BaseModel):
    type: AlignmentType
    origin: AlignmentGroupOrigin
    similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_indices: list[int]
    target_indices: list[int]
    source_tokens: list[str]
    target_tokens: list[str]
    links: list[AlignmentLink]


class SentenceAlignment(BaseModel):
    index: int
    id: str | None
    source: str
    target: str
    source_tokens: list[Token]
    target_tokens: list[Token]
    links: list[AlignmentLink]
    alignment_groups: list[AlignmentGroup]
    unaligned_source_indices: list[int]
    unaligned_target_indices: list[int]


class AlignmentResponse(BaseModel):
    source_language: str
    target_language: str
    model: str
    embedding_layer: int
    method: AlignmentMethod
    confidence_method: ConfidenceMethod
    sentence_alignments: list[SentenceAlignment]


class SupportedLanguage(BaseModel):
    code: str
    name: str
    native_name: str
    tokenizer: WordTokenizerType


class SupportedLanguagesResponse(BaseModel):
    model: str
    pairing: Literal["any-to-any"] = "any-to-any"
    total: int
    languages: list[SupportedLanguage]


class HealthResponse(BaseModel):
    status: Literal["ok", "ready"]
    model: str | None = None
    model_loaded: bool | None = None
    load_mode: Literal["eager", "lazy"] | None = None
