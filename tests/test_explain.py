import json

import pytest

from whoosh import fields, query, scoring
from whoosh.filedb.filestore import RamStorage
from whoosh.query import Wildcard
from whoosh.scoring import bm25

TOLERANCE = 1e-9


def make_index(docs, merge=True, **field_kwargs):
    schema = fields.Schema(
        title=fields.TEXT(stored=True), content=fields.TEXT(stored=True)
    )
    ix = RamStorage().create_index(schema)
    if merge:
        w = ix.writer()
        for title, content in docs:
            w.add_document(title=title, content=content)
        w.commit()
    else:
        for title, content in docs:
            w = ix.writer()
            w.add_document(title=title, content=content)
            w.commit(merge=False)
    return ix


def search_scores(searcher, q):
    results = searcher.search(q, limit=None)
    return {results.docnum(i): results.score(i) for i in range(len(results))}


def assert_matches_search(searcher, q, docnums=None):
    scores = search_scores(searcher, q)
    if docnums is None:
        docnums = scores
    for docnum in docnums:
        ex = searcher.explain(q, docnum)
        assert ex["matched"]
        assert ex["docnum"] == docnum
        assert abs(ex["score"] - scores[docnum]) < TOLERANCE
    return scores


def recompute(node):
    # Recomputes a node's score from the contributions listed in the tree
    assert node["matched"]
    if "children" in node:
        vals = [recompute(child) for child in node["children"] if child["matched"]]
        if node["combine"] == "max":
            combined = max(vals)
        else:
            assert node["combine"] == "sum"
            combined = sum(vals)
    else:
        d = node["details"]
        assert d["model"] == "BM25F"
        combined = bm25(
            d["idf"],
            d["tf"],
            d["field_length"],
            d["avg_field_length"],
            d["B"],
            d["K1"],
        )
    return combined * node["boost"]


def assert_recomputable(node):
    assert abs(recompute(node) - node["score"]) < TOLERANCE


def test_term_leaf_details():
    ix = make_index(
        [
            ("a", "alfa bravo charlie"),
            ("b", "alfa"),
            ("c", "alfa alfa alfa bravo charlie delta echo"),
        ]
    )
    with ix.searcher() as s:
        q = query.Term("content", "alfa")
        assert_matches_search(s, q)

        # Different field lengths and term frequencies show up in the leaves
        ex0 = s.explain(q, 0)
        ex2 = s.explain(q, 2)
        for ex in (ex0, ex2):
            assert ex["type"] == "Term"
            assert ex["supported"]
            d = ex["details"]
            assert d["model"] == "BM25F"
            assert d["B"] == 0.75
            assert d["K1"] == 1.2
            assert d["avg_field_length"] == s.avg_field_length("content")
            assert d["idf"] == s.idf("content", "alfa")

        assert ex0["details"]["tf"] == 1.0
        assert ex0["details"]["field_length"] == 3
        # Repeated terms increase tf
        assert ex2["details"]["tf"] == 3.0
        assert ex2["details"]["field_length"] == 7

        assert_recomputable(ex0)
        assert_recomputable(ex2)


def test_term_boost():
    ix = make_index([("a", "alfa bravo"), ("b", "alfa")])
    with ix.searcher() as s:
        q = query.Term("content", "alfa", boost=2.5)
        assert_matches_search(s, q)
        ex = s.explain(q, 0)
        assert ex["boost"] == 2.5
        assert_recomputable(ex)
        # The boost multiplies the raw BM25F score
        d = ex["details"]
        raw = bm25(
            d["idf"],
            d["tf"],
            d["field_length"],
            d["avg_field_length"],
            d["B"],
            d["K1"],
        )
        assert abs(ex["score"] - raw * 2.5) < TOLERANCE


def test_and():
    ix = make_index([("a", "alfa bravo charlie"), ("b", "alfa"), ("c", "bravo alfa")])
    with ix.searcher() as s:
        q = query.And([query.Term("content", "alfa"), query.Term("content", "bravo")])
        assert_matches_search(s, q)
        ex = s.explain(q, 0)
        assert ex["combine"] == "sum"
        assert len(ex["children"]) == 2
        assert all(child["matched"] for child in ex["children"])
        assert_recomputable(ex)


def test_or_partial_match():
    ix = make_index([("a", "alfa bravo charlie"), ("b", "alfa"), ("c", "bravo delta")])
    q = query.Or([query.Term("content", "alfa"), query.Term("content", "delta")])
    with ix.searcher() as s:
        assert_matches_search(s, q)

        # Document 0 only matches the "alfa" child
        ex = s.explain(q, 0)
        assert ex["combine"] == "sum"
        matched = [c for c in ex["children"] if c["matched"]]
        unmatched = [c for c in ex["children"] if not c["matched"]]
        assert len(matched) == 1
        assert len(unmatched) == 1
        # The non-matching child is not a fake zero-score explanation
        assert "score" not in unmatched[0]
        assert unmatched[0]["reason"] == "no_match"
        assert_recomputable(ex)


def test_phrase_boost():
    ix = make_index(
        [
            ("a", "alfa bravo charlie delta"),
            ("b", "bravo charlie"),
            ("c", "alfa charlie bravo"),
        ]
    )
    q = query.Phrase("content", ["alfa", "bravo"], boost=2.5)
    with ix.searcher() as s:
        assert_matches_search(s, q)
        ex = s.explain(q, 0)
        assert ex["type"] == "Phrase"
        assert ex["boost"] == 2.5
        assert ex["combine"] == "sum"
        assert ex["details"]["slop"] == 1
        assert ex["details"]["words"] == ["alfa", "bravo"]
        assert [child["term"] for child in ex["children"]] == ["alfa", "bravo"]
        assert_recomputable(ex)


def test_phrase_no_match():
    ix = make_index([("a", "alfa charlie bravo"), ("b", "alfa bravo")])
    q = query.Phrase("content", ["alfa", "bravo"])
    with ix.searcher() as s:
        # Both words are in document 0, but not as a phrase
        ex = s.explain(q, 0)
        assert not ex["matched"]
        assert ex["reason"] == "phrase_no_match"
        assert "score" not in ex
        assert "children" not in ex


def test_disjunction_max_tiebreak():
    ix = make_index(
        [
            ("alfa", "alfa bravo charlie"),
            ("bravo", "alfa"),
            ("charlie", "delta"),
        ]
    )
    q = query.DisjunctionMax(
        [query.Term("title", "alfa"), query.Term("content", "alfa")],
        tiebreak=0.3,
    )
    with ix.searcher() as s:
        assert_matches_search(s, q)
        ex = s.explain(q, 0)
        assert ex["combine"] == "max"
        assert ex["details"]["tiebreak"] == 0.3
        # The score is the maximum of the matching children's scores
        child_scores = [c["score"] for c in ex["children"] if c["matched"]]
        assert abs(ex["score"] - max(child_scores) * ex["boost"]) < TOLERANCE
        assert_recomputable(ex)


def test_deleted_document():
    ix = make_index([("a", "alfa bravo"), ("b", "alfa"), ("c", "bravo alfa")])
    w = ix.writer()
    w.delete_document(1)
    w.commit()

    q = query.Term("content", "alfa")
    with ix.searcher() as s:
        ex = s.explain(q, 1)
        assert not ex["matched"]
        assert ex["reason"] == "deleted"
        assert "score" not in ex
        assert "children" not in ex

        # Other documents still explain normally
        assert_matches_search(s, q)


def test_filter_and_mask():
    ix = make_index([("a", "alfa bravo"), ("b", "alfa"), ("c", "bravo alfa")])
    q = query.Term("content", "alfa")
    with ix.searcher() as s:
        ex = s.explain(q, 0, filter={1, 2})
        assert not ex["matched"]
        assert ex["reason"] == "filtered"
        assert "score" not in ex

        ex = s.explain(q, 0, filter=query.Term("content", "delta"))
        assert not ex["matched"]
        assert ex["reason"] == "filtered"

        ex = s.explain(q, 0, mask={0})
        assert not ex["matched"]
        assert ex["reason"] == "masked"

        # A filter that allows the document doesn't change the explanation
        ex = s.explain(q, 0, filter={0, 1})
        assert ex["matched"]
        assert_matches_search(s, q)


def test_multisegment_global_docnums():
    docs = [
        ("t0", "alfa bravo"),
        ("t1", "charlie"),
        ("t2", "bravo alfa alfa"),
        ("t3", "delta alfa"),
        ("t4", "echo alfa"),
    ]
    ix = make_index(docs, merge=False)
    with ix.searcher() as s:
        assert not s.is_atomic()
        assert len(s.leaf_searchers()) > 1

        q = query.Or([query.Term("content", "alfa"), query.Term("content", "bravo")])
        scores = assert_matches_search(s, q)

        # Documents in later segments keep their global identity
        for docnum in scores:
            ex = s.explain(q, docnum)
            assert ex["docnum"] == docnum
            assert_recomputable(ex)


def test_statistics_snapshot_after_update():
    ix = make_index([("a", "alfa bravo"), ("b", "alfa"), ("c", "bravo alfa")])
    q = query.Term("content", "alfa")

    s = ix.searcher()
    try:
        before = search_scores(s, q)
        explanations = {d: s.explain(q, d)["score"] for d in before}

        # Update the index after opening the searcher
        w = ix.writer()
        w.add_document(title="d", content="alfa alfa alfa alfa")
        w.commit()
        assert not s.up_to_date()

        # The old searcher still uses its own statistics snapshot, for both
        # searching and explaining
        assert search_scores(s, q) == before
        for docnum, score in explanations.items():
            assert s.explain(q, docnum)["score"] == score
    finally:
        s.close()

    # A new searcher sees the updated statistics
    with ix.searcher() as s2:
        after = search_scores(s2, q)
        assert len(after) == len(before) + 1
        assert after[0] != before[0]
        assert_matches_search(s2, q)


def test_custom_weighting_unsupported():
    class MyWeighting(scoring.WeightingModel):
        def scorer(self, searcher, fieldname, text, qf=1):
            return scoring.WeightScorer.for_(searcher, fieldname, text)

    ix = make_index([("a", "alfa bravo"), ("b", "alfa")])
    q = query.Term("content", "alfa")
    with ix.searcher(weighting=MyWeighting()) as s:
        ex = s.explain(q, 0)
        # The real score is still reported, but the leaf is marked as
        # unsupported instead of faking a breakdown
        assert ex["matched"]
        assert not ex["supported"]
        assert ex["reason"] == "unsupported_weighting_model"
        assert ex["details"] == {"model": "MyWeighting"}
        assert_matches_search(s, q)


def test_custom_weighting_with_explain():
    class MyWeighting(scoring.BM25F):
        def explain(self, searcher, fieldname, text, matcher):
            details = super().explain(searcher, fieldname, text, matcher)
            details["custom"] = True
            return details

    ix = make_index([("a", "alfa bravo"), ("b", "alfa")])
    q = query.Term("content", "alfa")
    with ix.searcher(weighting=MyWeighting()) as s:
        ex = s.explain(q, 0)
        assert ex["supported"]
        assert ex["details"]["custom"]
        assert_matches_search(s, q)


def test_unsupported_query_type():
    ix = make_index([("a", "alfa bravo"), ("b", "alpine")])
    with ix.searcher() as s:
        ex = s.explain(Wildcard("content", "al*"), 0)
        assert ex["matched"]
        assert not ex["supported"]
        assert ex["reason"] == "unsupported_query"


def test_serializable():
    ix = make_index([("alfa", "alfa bravo charlie"), ("bravo", "alfa"), ("c", "delta")])
    queries = [
        query.Term("content", "alfa"),
        query.And([query.Term("content", "alfa"), query.Term("title", "alfa")]),
        query.Or([query.Term("content", "alfa"), query.Term("content", "delta")]),
        query.DisjunctionMax(
            [query.Term("title", "alfa"), query.Term("content", "alfa")],
            tiebreak=0.2,
        ),
        query.Phrase("content", ["alfa", "bravo"], boost=2.0),
    ]
    with ix.searcher() as s:
        for q in queries:
            for docnum in search_scores(s, q):
                json.dumps(s.explain(q, docnum))
        # Failure explanations are serializable too
        json.dumps(s.explain(query.Term("content", "delta"), 0))


def test_docnum_out_of_range():
    ix = make_index([("a", "alfa bravo")])
    with ix.searcher() as s:
        with pytest.raises(ValueError):
            s.explain(query.Term("content", "alfa"), 5)
        with pytest.raises(ValueError):
            s.explain(query.Term("content", "alfa"), -1)


def test_or_with_scale():
    ix = make_index([("a", "alfa bravo charlie"), ("b", "alfa"), ("c", "bravo alfa")])
    q = query.Or(
        [
            query.Term("content", "alfa"),
            query.Term("content", "bravo"),
            query.Term("content", "charlie"),
        ],
        scale=0.5,
    )
    with ix.searcher() as s:
        assert_matches_search(s, q)
        ex = s.explain(q, 0)
        assert ex["details"]["scale"] == 0.5
        assert "coord" in ex["details"]
