from whoosh import fields, spelling
from whoosh.spelling import (
    EditCosts,
    Lexicon,
    SegmentCorrector,
    tokenize_query_string,
    weighted_distance,
)
from whoosh.util.testing import TempIndex

WORDS = (
    "new york hotel hotels grand city newark plaza inn suites manhattan brooklyn"
).split()

BIGRAMS = {
    ("new", "york"): 100,
    ("york", "hotel"): 60,
    ("grand", "hotel"): 40,
}


def make_lexicon():
    return Lexicon(WORDS, dict.fromkeys(WORDS, 10))


def make_corrector(**kwargs):
    kwargs.setdefault("bigram_freq", BIGRAMS)
    return SegmentCorrector(make_lexicon(), **kwargs)


def test_weighted_distance():
    costs = EditCosts(insert=1, delete=2, replace=3)
    assert weighted_distance("abc", "abc", costs) == 0
    assert weighted_distance("abc", "ab", costs) == 2  # delete
    assert weighted_distance("ab", "abc", costs) == 1  # insert
    assert weighted_distance("abc", "abd", costs) == 3  # replace
    # Early cutoff returns something over the budget
    assert weighted_distance("abc", "xyz", costs, max_cost=1) > 1


def test_lexicon_from_reader_reads_dictionary():
    schema = fields.Schema(text=fields.TEXT)
    with TempIndex(schema, "seglexicon") as ix:
        with ix.writer() as w:
            w.add_document(text="new york hotel")
            w.add_document(text="new york city")
            w.add_document(text="grand hotel")

        with ix.reader() as r:
            lex = Lexicon.from_reader(r, "text")

        assert set(lex.words()) == {"new", "york", "hotel", "city", "grand"}
        assert lex.frequency("hotel") == 2
        assert lex.frequency("new") == 2
        assert lex.frequency("grand") == 1
        assert "hotel" in lex
        assert "hotle" not in lex


def test_split_and_spell_combined():
    sc = make_corrector()
    res = sc.correct_string("newyork hotle")

    assert res.exhaustive
    assert res.text == "new york hotel"

    edits = res.edits
    assert len(edits) == 2
    # Each edit carries the original span, the replacement, and a reason
    assert (edits[0].start, edits[0].end) == (0, 7)
    assert edits[0].original == "newyork"
    assert edits[0].replacement == "new york"
    assert edits[0].reason == "split"
    assert (edits[1].start, edits[1].end) == (8, 13)
    assert edits[1].original == "hotle"
    assert edits[1].replacement == "hotel"
    assert edits[1].reason == "spell"

    # The rewritten text is the original string with the edit spans applied
    original = "newyork hotle"
    out, pos = [], 0
    for e in edits:
        out.append(original[pos : e.start])
        out.append(e.replacement)
        pos = e.end
    out.append(original[pos:])
    assert "".join(out) == res.text


def test_split_with_spelling_on_part():
    sc = make_corrector()
    res = sc.correct_string("newyrok")
    assert res.text == "new york"
    assert res.edits[0].reason == "spell+split"
    assert res.edits[0].replacement == "new york"


def test_merge_two_words():
    sc = make_corrector()
    res = sc.correct_string("hot el")
    assert res.text == "hotel"
    assert len(res.edits) == 1
    edit = res.edits[0]
    assert (edit.start, edit.end) == (0, 6)
    assert edit.original == "hot el"
    assert edit.replacement == "hotel"
    assert edit.reason == "merge"


def test_quoted_phrase_preserved():
    sc = make_corrector()
    res = sc.correct_string('"newyork hotle" hotle')
    assert res.text == '"newyork hotle" hotel'
    # The only edit is outside the quoted span
    assert len(res.edits) == 1
    assert res.edits[0].start == 16
    assert res.edits[0].reason == "spell"


def test_wildcard_and_field_syntax_preserved():
    sc = make_corrector()

    res = sc.correct_string("sku:A* hotle")
    assert res.text == "sku:A* hotel"
    assert len(res.edits) == 1
    assert res.edits[0].original == "hotle"

    # The field prefix is kept, the word after it may be corrected
    res = sc.correct_string("title:hotle")
    assert res.text == "title:hotel"
    assert res.edits[0].start == 6


def test_boolean_operators_preserved():
    sc = make_corrector()
    res = sc.correct_string("newyork AND hotle OR inn")
    assert res.text == "new york AND hotel OR inn"
    assert [e.reason for e in res.edits] == ["split", "spell"]


def test_tokenize_query_string():
    tokens = tokenize_query_string('sku:A* "hotle inn" title:hotle AND hotel')
    kinds = [t[0] for t in tokens]
    assert kinds == ["syntax", "syntax", "syntax", "term", "syntax", "term"]
    assert tokens[2][1] == "title:"
    assert tokens[3][1] == "hotle"
    assert tokens[5][1] == "hotel"


def test_context_changes_ranking():
    lex = Lexicon(["new", "hotel", "hotels"], {"new": 10, "hotel": 10, "hotels": 10})

    # Without context, the cheaper correction wins
    sc = SegmentCorrector(lex, bigram_freq=None)
    assert sc.correct_string("new hotle").text == "new hotel"

    # A strong adjacent-word frequency can flip the ranking
    sc = SegmentCorrector(lex, bigram_freq={("new", "hotels"): 1000})
    assert sc.correct_string("new hotle").text == "new hotels"


def test_configurable_costs_disable_split_by_price():
    # A very expensive split operation prevents splitting
    sc = make_corrector(costs=EditCosts(split=100.0))
    res = sc.correct_string("newyork hotle")
    assert "new york" not in res.text
    assert all(e.reason != "split" for e in res.edits)
    # "hotle" is still spell-corrected
    assert res.text.endswith("hotel")


def test_top_k_and_deterministic_order():
    sc = make_corrector()
    res1 = sc.correct_string("newyork hotle", k=5)
    res2 = sc.correct_string("newyork hotle", k=5)

    assert 1 < len(res1.rewrites) <= 5
    # Scores are non-decreasing and the order is deterministic
    scores = [rw.cost - rw.context for rw in res1.rewrites]
    assert scores == sorted(scores)
    assert [rw.text for rw in res1.rewrites] == [rw.text for rw in res2.rewrites]


def test_state_limit_marks_not_exhaustive():
    sc = make_corrector(max_states=1)
    res = sc.correct_string("newyork hotle")
    assert not res.exhaustive


def test_candidate_limit_marks_not_exhaustive():
    # "hotell" has at least two corrections within the default cost budget
    sc = make_corrector(max_candidates=1)
    res = sc.correct_string("hotell")
    assert not res.exhaustive

    sc = make_corrector()
    res = sc.correct_string("hotell")
    assert res.exhaustive


def test_per_word_behavior_preserved_without_split_merge():
    lex = make_lexicon()
    sc = SegmentCorrector(lex, allow_split=False, allow_merge=False)

    # A single typo is corrected to the best dictionary word
    res = sc.correct_string("hotle")
    assert res.text == "hotel"
    assert [e.reason for e in res.edits] == ["spell"]

    # No split or merge edits are ever produced
    res = sc.correct_string("newyork hotle")
    assert all(e.reason == "spell" for e in res.edits)
    assert "new york" not in res.text

    # word_suggestions mirrors the classic Corrector.suggest behavior: same
    # candidate set as ListCorrector over the same word list
    wordlist = sorted(WORDS)
    classic = spelling.ListCorrector(wordlist)
    for typo in ("hotell", "yokr", "newy"):
        maxdist = 2
        expected = set(classic.suggest(typo, maxdist=maxdist))
        got = set(sc.word_suggestions(typo, limit=10, max_cost=maxdist))
        assert got == expected


def test_word_suggestions_ordering():
    # Equal frequencies: ties break by dictionary order
    lex = Lexicon(["hotel", "hotels"], {"hotel": 5, "hotels": 5})
    sc = SegmentCorrector(lex)
    assert sc.word_suggestions("hotell") == ["hotel", "hotels"]

    # Higher frequency wins ties
    lex = Lexicon(["hotel", "hotels"], {"hotel": 5, "hotels": 50})
    sc = SegmentCorrector(lex)
    assert sc.word_suggestions("hotell") == ["hotels", "hotel"]


def test_suggest_segment_plain_text():
    sc = make_corrector()
    res = sc.suggest_segment("newyork hotle")
    assert res.text == "new york hotel"
    assert res.original == "newyork hotle"


def test_index_backed_end_to_end():
    schema = fields.Schema(text=fields.TEXT, sku=fields.ID)
    with TempIndex(schema, "segcorrector") as ix:
        with ix.writer() as w:
            w.add_document(text="new york hotel grand", sku="A1")
            w.add_document(text="new york city hotels", sku="A2")
            w.add_document(text="grand hotel plaza inn", sku="B1")

        with ix.reader() as r:
            lex = Lexicon.from_reader(r, "text")

        sc = SegmentCorrector(lex, bigram_freq=BIGRAMS)
        res = sc.correct_string("newyork hotle")
        assert res.text == "new york hotel"

        # Unmodifiable wildcard term is preserved verbatim
        res = sc.correct_string("sku:A* newyork")
        assert res.text == "sku:A* new york"


def test_no_change_for_clean_query():
    sc = make_corrector()
    res = sc.correct_string("new york hotel")
    assert res.text == "new york hotel"
    assert res.edits == []
