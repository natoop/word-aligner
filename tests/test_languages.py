from app.languages import SUPPORTED_LANGUAGES


def test_supported_language_codes_are_unique() -> None:
    codes = [language.code for language in SUPPORTED_LANGUAGES]

    assert len(codes) == len(set(codes))


def test_chinese_variants_use_the_dedicated_tokenizer() -> None:
    tokenizers = {language.code: language.tokenizer for language in SUPPORTED_LANGUAGES}

    assert tokenizers["zh-Hans"] == "jieba"
    assert tokenizers["zh-Hant"] == "jieba"
