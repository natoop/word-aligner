from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

_Language = Annotated[str, StringConstraints(pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$", max_length=35)]


class LLMTextPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=100)
    source: str = Field(min_length=1, max_length=10_000)
    target: str = Field(min_length=1, max_length=10_000)

    @field_validator("source", "target")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain at least one non-whitespace character")
        return value


class LLMAlignmentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "provider": "deepseek",
                "source_language": "en",
                "target_language": "zh-Hans",
                "sentence_pairs": [{"id": "example", "source": "He gave up.", "target": "他放弃了。"}],
            }
        },
    )

    provider: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    source_language: _Language = "auto"
    target_language: _Language = "auto"
    sentence_pairs: list[LLMTextPair] = Field(min_length=1, max_length=10)


class LLMProviderInfo(BaseModel):
    id: str
    name: str
    model: str


class LLMProvidersResponse(BaseModel):
    providers: list[LLMProviderInfo]
    default_provider: str | None


class LLMToken(BaseModel):
    index: int
    text: str
    start: int = Field(ge=0, description="Inclusive UTF-16 code-unit offset into the original text")
    end: int = Field(ge=1, description="Exclusive UTF-16 code-unit offset into the original text")
    is_protected: bool = False


class LLMAlignmentGroup(BaseModel):
    type: Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
    origin: Literal["llm"] = "llm"
    similarity: None = None
    confidence: None = None
    source_indices: list[int]
    target_indices: list[int]
    source_tokens: list[str]
    target_tokens: list[str]
    links: list[dict[str, object]] = Field(default_factory=list, max_length=0)


class LLMSentenceAlignment(BaseModel):
    index: int
    id: str | None
    source: str
    target: str
    source_tokens: list[LLMToken]
    target_tokens: list[LLMToken]
    links: list[dict[str, object]] = Field(default_factory=list, max_length=0)
    alignment_groups: list[LLMAlignmentGroup]
    unaligned_source_indices: list[int]
    unaligned_target_indices: list[int]
    usage: dict[str, int] | None = None


class LLMAlignmentResponse(BaseModel):
    source_language: str
    target_language: str
    provider: str
    provider_name: str
    model: str
    method: Literal["llm"] = "llm"
    offset_unit: Literal["utf16"] = "utf16"
    confidence_method: None = None
    usage: dict[str, int] | None = Field(
        default=None,
        description="Summed provider-reported token usage; null if any sentence omitted usage",
    )
    sentence_alignments: list[LLMSentenceAlignment]
