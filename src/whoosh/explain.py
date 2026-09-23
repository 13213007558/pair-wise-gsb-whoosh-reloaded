"""
Read-only score explanation support.

This module implements the machinery behind
:meth:`whoosh.searching.Searcher.explain`, which returns a
JSON-serializable tree of dictionaries describing how the score of a
document was computed for a query.

Every node in the tree is a plain dictionary with at least the keys
``type``, ``matched``, and ``score``. Nodes for documents that
did not match (including documents rejected by a filter, deleted
documents, and documents where a phrase did not match) have
``matched`` set to False, ``score`` set to None, and a
``reason`` key, instead of a normal-looking zero score.

All scores in the tree are produced by the same matcher/scorer objects
used by a real search on the same searcher, so the top-level score is
guaranteed to be consistent with the score returned by
:meth:`whoosh.searching.Searcher.search` (within floating point
tolerance), and the contributions listed in each parent node can be
recombined (summed, maximized, or multiplied by a boost, as described by
the node's ``description``) to recompute the final score.
"""

from whoosh.query.compound import And, DisjunctionMax, Or
from whoosh.query.positional import Phrase
from whoosh.query.terms import Term


def _display(text):
    # Converts term text to a JSON-friendly string for display
    if isinstance(text, bytes):
        try:
            return text.decode("ascii")
        except UnicodeDecodeError:
            return repr(text)
    return text


def _query_string(q):
    try:
        return str(q)
    except Exception:
        return repr(q)


def _weighting_for(searcher, context):
    # Mirrors the weighting selection logic in Term.matcher/postings
    w = context.weighting if context else None
    return w or searcher.weighting


def _score_query(searcher, q, docnum, context):
    # Builds the real matcher for the query and positions it on the
    # document. Returns a (matched, score, matcher) tuple. The score is
    # exactly the score a real search would compute for this document.
    m = q.matcher(searcher, context)
    if not m.is_active():
        return False, None, m
    if m.id() < docnum:
        m.skip_to(docnum)
    if not m.is_active() or m.id() != docnum:
        return False, None, m
    return True, m.score(), m


def explain_query(searcher, q, docnum, context=None):
    """Returns a JSON-serializable dictionary tree explaining the score of
    the given query for the given document number (in the searcher's
    global numbering).

    This function does not check whether the document is deleted or
    excluded by a filter; use :meth:`whoosh.searching.Searcher.explain`
    for the full read-only entry point.
    """

    boost = getattr(q, "boost", 1.0)
    if boost != 1.0:
        return _explain_boost(searcher, q, boost, docnum, context)
    if isinstance(q, Term):
        return _explain_term(searcher, q, docnum, context)
    if isinstance(q, Phrase):
        return _explain_phrase(searcher, q, docnum, context)
    if isinstance(q, (And, Or, DisjunctionMax)):
        return _explain_compound(searcher, q, docnum, context)
    return _explain_generic(searcher, q, docnum, context)


def _explain_boost(searcher, q, boost, docnum, context):
    child = explain_query(searcher, q.with_boost(1.0), docnum, context)
    matched, score, _ = _score_query(searcher, q, docnum, context)
    node = {
        "type": "boost",
        "boost": boost,
        "description": "the child score is multiplied by the boost factor",
        "matched": matched,
        "score": score if matched else None,
        "children": [child],
    }
    if not matched:
        node["reason"] = "the boosted query did not match the document"
    return node


def _explain_term(searcher, q, docnum, context):
    fieldname = q.fieldname
    text = q.text
    node = {
        "type": "term",
        "field": fieldname,
        "term": _display(text),
        "matched": False,
        "score": None,
    }

    if fieldname not in searcher.schema:
        node["reason"] = f"field {fieldname!r} is not in the schema"
        return node

    field = searcher.schema[fieldname]
    try:
        btext = field.to_bytes(text)
    except ValueError:
        node["reason"] = "the term text could not be converted to bytes"
        return node

    if (fieldname, btext) not in searcher.reader():
        node["reason"] = "the term is not in the index"
        return node

    matched, score, m = _score_query(searcher, q, docnum, context)
    if not matched:
        node["reason"] = "the term does not appear in the document"
        return node

    node["matched"] = True
    node["score"] = score
    # The term frequency and field length actually used by the scorer
    node["tf"] = m.weight()
    node["doc_length"] = searcher.doc_field_length(docnum, fieldname, 1)

    weighting = _weighting_for(searcher, context)
    details = weighting.explain(searcher, fieldname, btext, docnum, m)
    if details is None:
        node["model"] = type(weighting).__name__
        node["supported"] = False
        node["note"] = (
            f"weighting model {type(weighting).__name__} does not implement "
            "the explain protocol"
        )
    else:
        node["supported"] = True
        node.update(details)
    return node


def _explain_phrase(searcher, q, docnum, context):
    fieldname = q.fieldname
    node = {
        "type": "phrase",
        "field": fieldname,
        "words": [_display(word) for word in q.words],
        "slop": q.slop,
        "matched": False,
        "score": None,
    }

    if fieldname not in searcher.schema:
        node["reason"] = f"field {fieldname!r} is not in the schema"
        return node

    field = searcher.schema[fieldname]
    if not field.format or not field.format.supports("positions"):
        node["reason"] = f"field {fieldname!r} does not store positions"
        return node

    reader = searcher.reader()
    missing = []
    for word in q.words:
        try:
            bword = field.to_bytes(word)
        except ValueError:
            missing.append(word)
            continue
        if (fieldname, bword) not in reader:
            missing.append(word)
    if missing:
        node["reason"] = "words not in the index: " + ", ".join(
            repr(_display(word)) for word in missing
        )
        return node

    matched, score, _ = _score_query(searcher, q, docnum, context)
    if not matched:
        node["reason"] = "the words do not appear as a phrase in the document"
        return node

    node["matched"] = True
    node["score"] = score
    node["description"] = "sum of the scores of the phrase words"
    node["children"] = [
        explain_query(searcher, Term(fieldname, word), docnum, context)
        for word in q.words
    ]
    return node


def _explain_compound(searcher, q, docnum, context):
    if isinstance(q, And):
        qtype = "and"
        description = "sum of the sub-query scores; all sub-queries must match"
    elif isinstance(q, DisjunctionMax):
        qtype = "dismax"
        description = "maximum of the matching sub-query scores"
    else:
        qtype = "or"
        description = "sum of the matching sub-query scores"

    children = [
        explain_query(searcher, subq, docnum, context) for subq in q.subqueries
    ]
    matched, score, _ = _score_query(searcher, q, docnum, context)
    node = {
        "type": qtype,
        "description": description,
        "matched": matched,
        "score": score if matched else None,
        "children": children,
    }

    params = {}
    if isinstance(q, DisjunctionMax):
        params["tiebreak"] = q.tiebreak
    if isinstance(q, Or) and q.scale:
        params["scale"] = q.scale
        node["note"] = "coord scaling is applied on top of the child scores"
    if params:
        node["params"] = params

    if not matched:
        if isinstance(q, And):
            node["reason"] = "not all sub-queries matched the document"
        else:
            node["reason"] = "no sub-query matched the document"
    return node


def _explain_generic(searcher, q, docnum, context):
    matched, score, _ = _score_query(searcher, q, docnum, context)
    node = {
        "type": "unsupported",
        "query_type": q.__class__.__name__,
        "query": _query_string(q),
        "matched": matched,
        "score": score if matched else None,
        "note": (
            f"explain does not support {q.__class__.__name__} queries; "
            "the score was computed by the real matcher"
        ),
    }
    if not matched:
        node["reason"] = "the query did not match the document"
    return node
