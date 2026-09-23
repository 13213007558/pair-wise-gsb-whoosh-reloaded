import pytest

from whoosh import analysis, fields, query, scoring
from whoosh.filedb.filestore import RamStorage

EN_NAME = "test_ml_en"
ZH_NAME = "test_ml_zh"


def make_en_analyzer():
    # Word-based English-ish chain: split on non-letters, lowercase,
    # remove stop words
    return (
        analysis.RegexTokenizer() | analysis.LowercaseFilter() | analysis.StopFilter()
    )


def make_zh_analyzer():
    # A deliberately different, simple Chinese-ish chain: overlapping
    # character bigrams, no word segmentation service required
    return analysis.NgramTokenizer(2)


@pytest.fixture
def registry():
    analysis.register_analyzer(EN_NAME, make_en_analyzer(), version=1)
    analysis.register_analyzer(ZH_NAME, make_zh_analyzer(), version=1)
    yield
    analysis.unregister_analyzer(EN_NAME)
    analysis.unregister_analyzer(ZH_NAME)


def make_schema(**field_kwargs):
    field_kwargs.setdefault("stored", True)
    return fields.Schema(
        id=fields.ID(stored=True, unique=True),
        content=fields.MultiLangTEXT(
            {"en": EN_NAME, "zh": ZH_NAME}, default_lang="en", **field_kwargs
        ),
    )


def make_index(schema):
    ix = RamStorage().create_index(schema)
    return ix


def add_two_docs(ix):
    writer = ix.writer()
    writer.add_document(id="1", content="the quick brown fox", _lang_content="en")
    writer.add_document(
        id="2",
        content="\u6211\u7231\u5317\u4eac\u5929\u5b89\u95e8",
        _lang_content="zh",
    )
    writer.commit()


def ids(results):
    return sorted(hit["id"] for hit in results)


def test_two_languages_one_field(registry):
    schema = make_schema()
    ix = make_index(schema)
    add_two_docs(ix)
    field = schema["content"]

    with ix.searcher() as s:
        q = field.query("content", "quick", langs=["en"])
        assert ids(s.search(q)) == ["1"]

        q = field.query("content", "\u5317\u4eac", langs=["zh"])
        assert ids(s.search(q)) == ["2"]

        # The English word analyzed by the Chinese branch (bigrams) does not
        # hit anything, and vice versa
        q = field.query("content", "quick", langs=["zh"])
        assert ids(s.search(q)) == []

        q = field.query("content", "\u5317\u4eac", langs=["en"])
        assert ids(s.search(q)) == []

        # No language selection: branches for all configured languages
        q = field.query("content", "quick")
        assert ids(s.search(q)) == ["1"]
        q = field.query("content", "\u5317\u4eac")
        assert ids(s.search(q)) == ["2"]


def test_terms_keep_analysis_chain_identity(registry):
    # Both analyzers emit the identical token "ab" for their document, but
    # the indexed terms must not collide across languages
    schema = make_schema()
    ix = make_index(schema)
    writer = ix.writer()
    writer.add_document(id="1", content="ab cd", _lang_content="en")
    writer.add_document(id="2", content="ab\u5317\u4eac", _lang_content="zh")
    writer.commit()
    field = schema["content"]

    with ix.searcher() as s:
        terms = set(s.reader().lexicon("content"))
        assert b"en\x01ab" in terms
        assert b"zh\x01ab" in terms

        q = field.query("content", "ab", langs=["en"])
        assert ids(s.search(q)) == ["1"]

        q = field.query("content", "ab", langs=["zh"])
        assert ids(s.search(q)) == ["2"]


def test_default_language(registry):
    schema = make_schema()
    ix = make_index(schema)
    writer = ix.writer()
    # No _lang_content: the field's default_lang decides
    writer.add_document(id="1", content="quick fox")
    writer.commit()
    field = schema["content"]

    with ix.searcher() as s:
        terms = set(s.reader().lexicon("content"))
        assert b"en\x01quick" in terms
        assert b"zh\x01quick" not in terms

        q = field.query("content", "quick", langs=["en"])
        assert ids(s.search(q)) == ["1"]


def test_unknown_language_falls_back_to_default(registry):
    schema = make_schema()
    ix = make_index(schema)
    writer = ix.writer()
    writer.add_document(id="1", content="quick fox", _lang_content="fr")
    writer.commit()

    with ix.searcher() as s:
        terms = set(s.reader().lexicon("content"))
        assert b"en\x01quick" in terms


def test_unknown_language_error():
    field = fields.MultiLangTEXT(
        {"en": make_en_analyzer()}, default_lang="en", on_unknown="error"
    )
    schema = fields.Schema(id=fields.ID(stored=True, unique=True), content=field)
    ix = make_index(schema)
    writer = ix.writer()
    with pytest.raises(fields.FieldConfigurationError):
        writer.add_document(id="1", content="quick fox", _lang_content="fr")
    writer.cancel()


def test_unknown_language_mapped_to_configured_language():
    field = fields.MultiLangTEXT(
        {"en": make_en_analyzer(), "zh": make_zh_analyzer()},
        default_lang="en",
        on_unknown="zh",
    )
    schema = fields.Schema(id=fields.ID(stored=True, unique=True), content=field)
    ix = make_index(schema)
    writer = ix.writer()
    writer.add_document(id="1", content="ab", _lang_content="fr")
    writer.commit()

    with ix.searcher() as s:
        terms = set(s.reader().lexicon("content"))
        assert b"zh\x01ab" in terms


def test_bad_configuration():
    with pytest.raises(fields.FieldConfigurationError):
        fields.MultiLangTEXT({})
    with pytest.raises(fields.FieldConfigurationError):
        fields.MultiLangTEXT({"en": make_en_analyzer()}, default_lang="fr")
    with pytest.raises(fields.FieldConfigurationError):
        fields.MultiLangTEXT({"en": make_en_analyzer()}, on_unknown="klingon")
    with pytest.raises(fields.FieldConfigurationError):
        fields.MultiLangTEXT({"en": "not_registered_name"})


def test_update_document_language_and_merge(registry):
    schema = make_schema()
    ix = make_index(schema)
    add_two_docs(ix)
    field = schema["content"]

    zh_word = "\u5317\u4eac"
    with ix.searcher() as s:
        q = field.query("content", zh_word, langs=["zh"])
        assert ids(s.search(q)) == ["2"]

    # Change the language of document 2 from Chinese to English and merge
    # all segments
    writer = ix.writer()
    writer.update_document(
        id="2",
        content="\u6211\u7231\u5317\u4eac\u5929\u5b89\u95e8",
        _lang_content="en",
    )
    writer.commit(optimize=True)

    with ix.searcher() as s:
        assert s.doc_count() == 2

        # The old language's terms were really removed by the merge
        terms = set(s.reader().lexicon("content"))
        assert "zh\x01\u5317\u4eac".encode("utf8") not in terms

        # The old language query no longer hits
        q = field.query("content", zh_word, langs=["zh"])
        assert ids(s.search(q)) == []

        # The document is now found through the English branch (the English
        # analyzer treats the Han characters as one word token)
        q = field.query(
            "content",
            "\u6211\u7231\u5317\u4eac\u5929\u5b89\u95e8",
            langs=["en"],
        )
        assert ids(s.search(q)) == ["2"]

        # And the English document is untouched
        q = field.query("content", "quick", langs=["en"])
        assert ids(s.search(q)) == ["1"]


def test_phrase_query_uses_branch_chain(registry):
    schema = make_schema()
    ix = make_index(schema)
    add_two_docs(ix)
    field = schema["content"]

    with ix.searcher() as s:
        q = field.phrase("content", "quick brown", langs=["en"])
        assert ids(s.search(q)) == ["1"]

        q = field.phrase("content", "brown quick", langs=["en"])
        assert ids(s.search(q)) == []

        # Chinese branch: the phrase is analyzed into bigrams and matched
        # positionally
        q = field.phrase("content", "\u5317\u4eac\u5929", langs=["zh"])
        assert ids(s.search(q)) == ["2"]

        q = field.phrase("content", "\u5317\u5929\u4eac", langs=["zh"])
        assert ids(s.search(q)) == []


def test_prefix_query_uses_branch_chain(registry):
    schema = make_schema()
    ix = make_index(schema)
    add_two_docs(ix)
    field = schema["content"]

    with ix.searcher() as s:
        q = field.prefix("content", "qui", langs=["en"])
        assert ids(s.search(q)) == ["1"]

        q = field.prefix("content", "QUI", langs=["en"])
        assert ids(s.search(q)) == ["1"]

        q = field.prefix("content", "\u5317", langs=["zh"])
        assert ids(s.search(q)) == ["2"]

        q = field.prefix("content", "qui", langs=["zh"])
        assert ids(s.search(q)) == []


def test_highlight_uses_matched_branch(registry):
    schema = make_schema()
    ix = make_index(schema)
    add_two_docs(ix)
    field = schema["content"]

    with ix.searcher() as s:
        # English hit highlighted through the English chain
        q = field.query("content", "quick", langs=["en"])
        results = s.search(q, terms=True)
        hit = results[0]
        assert hit["id"] == "1"
        text = hit.highlights("content")
        assert "<b" in text
        assert "quick" in text

        # Chinese hit: only the Chinese (bigram) chain can find the matched
        # term in the stored text, so a non-empty highlight proves the
        # matched branch's analysis chain was used
        q = field.query("content", "\u5317\u4eac", langs=["zh"])
        results = s.search(q, terms=True)
        hit = results[0]
        assert hit["id"] == "2"
        text = hit.highlights("content")
        assert "<b" in text
        assert "\u5317\u4eac" in text


def test_scoring_is_standard_bm25f(registry):
    schema = make_schema()
    ix = make_index(schema)
    add_two_docs(ix)
    field = schema["content"]

    with ix.searcher(weighting=scoring.BM25F()) as s:
        # The language-filtered query scores exactly like the public Term
        # query over the language-tagged term
        q = field.query("content", "quick", langs=["en"])
        results = s.search(q)
        assert len(results) == 1
        branch_score = results[0].score
        assert branch_score > 0

        explicit = query.Term("content", "en\x01quick")
        explicit_results = s.search(explicit)
        assert len(explicit_results) == 1
        assert explicit_results[0].score == pytest.approx(branch_score)

        # Widening the language set does not change the score of the
        # document matched by a single branch
        q = field.query("content", "quick", langs=["en", "zh"])
        results = s.search(q)
        assert len(results) == 1
        assert results[0].score == pytest.approx(branch_score)


def test_query_language_validation(registry):
    schema = make_schema()
    field = schema["content"]
    with pytest.raises(fields.FieldConfigurationError):
        field.query("content", "quick", langs=["fr"])
    with pytest.raises(fields.FieldConfigurationError):
        field.phrase("content", "quick", langs=["fr"])
    with pytest.raises(fields.FieldConfigurationError):
        field.prefix("content", "qui", langs=["fr"])


def test_registry_version_checked_on_open(registry):
    schema = make_schema()
    storage = RamStorage()
    ix = storage.create_index(schema)
    writer = ix.writer()
    writer.add_document(id="1", content="quick fox", _lang_content="en")
    writer.commit()

    # Same versions: the index opens fine
    ix2 = storage.open_index()
    assert ix2.doc_count() == 1

    # A new registered version of an analyzer must be reported clearly
    analysis.register_analyzer(EN_NAME, make_en_analyzer(), version=2)
    with pytest.raises(fields.FieldConfigurationError, match="version"):
        storage.open_index()

    # A missing analyzer registration must be reported clearly too
    analysis.unregister_analyzer(EN_NAME)
    with pytest.raises(fields.FieldConfigurationError, match=EN_NAME):
        storage.open_index()


def test_unregistered_analyzer_objects_not_version_checked():
    # Analyzer objects given directly (not by registered name) work and do
    # not participate in the registry version check
    field = fields.MultiLangTEXT(
        {"en": make_en_analyzer(), "zh": make_zh_analyzer()},
        default_lang="en",
    )
    schema = fields.Schema(id=fields.ID(stored=True, unique=True), content=field)
    storage = RamStorage()
    ix = storage.create_index(schema)
    writer = ix.writer()
    writer.add_document(id="1", content="quick fox", _lang_content="en")
    writer.commit()

    ix2 = storage.open_index()
    with ix2.searcher() as s:
        q = ix2.schema["content"].query("content", "quick", langs=["en"])
        assert ids(s.search(q)) == ["1"]


def test_plain_text_field_unaffected(registry):
    schema = fields.Schema(
        id=fields.ID(stored=True, unique=True),
        body=fields.TEXT(stored=True),
        content=fields.MultiLangTEXT(
            {"en": EN_NAME, "zh": ZH_NAME}, default_lang="en", stored=True
        ),
    )
    ix = make_index(schema)
    writer = ix.writer()
    writer.add_document(
        id="1",
        body="the quick brown fox",
        content="quick fox",
        _lang_content="en",
    )
    writer.add_document(
        id="2",
        body="lorem ipsum",
        content="\u5317\u4eac\u4e0a\u6d77",
        _lang_content="zh",
    )
    writer.commit()

    with ix.searcher() as s:
        q = query.Term("body", "quick")
        assert ids(s.search(q)) == ["1"]

        q = schema["content"].query("content", "\u5317\u4eac", langs=["zh"])
        assert ids(s.search(q)) == ["2"]
