# Copyright 2008 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT CHAPUT ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MATT CHAPUT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

"""
This module implements score explanations: a read-only view of how the score
of a particular document for a particular query was computed.

The entry point is :meth:`whoosh.searching.Searcher.explain`, which returns
a JSON-serializable tree of plain dicts. Each node in the tree corresponds to
a query object and has the following keys:

* `type`: the name of the query class (for example `"Term"`, `"And"`).
* `query`: the string representation of the (sub-)query.
* `matched`: whether this (sub-)query matched the document.
* `boost`: the boost applied at this node.
* `score`: the score of this node (only present if `matched` is True).
  A node's score is always `combine(children scores) * boost`, using the
  node's `combine` method (`"sum"` or `"max"`), so the contributions
  listed in the tree can be recomputed back into the final score.
* `supported`: whether this node (and all of its children) could be fully
  explained. Nodes that can't be fully explained include a `reason` key.
* `details`: a dict of node-specific information. Term leaves contain the
  actual statistics used by the weighting model (tf, field length, average
  field length, idf, and model parameters).
* `children`: a list of explanations for sub-queries.

If the document does not match, is deleted, or is rejected by a filter, the
top-level dict has `matched` set to False and a `reason` key, instead of
a misleading zero-score explanation.
"""

from whoosh import matching
from whoosh.query import And, DisjunctionMax, Or, Phrase, Term
from whoosh.reading import TermNotFound


def explain(searcher, q, docnum, filter=None, mask=None):
    """Returns a JSON-serializable explanation of the score of the document
    with the given number for the given query, using the statistics snapshot
    of the given searcher.

    :param searcher: the :class:`whoosh.searching.Searcher` to explain
        against. The `docnum` argument and the `docnum` key in the
        returned tree use this searcher's (global) document numbering, even
        when the index has multiple segments.
    :param q: the :class:`whoosh.query.Query` object to explain.
    :param docnum: the document number to explain, in the numbering of
        `searcher`.
    :param filter: an optional filter (a query, Results object, or set of
        docnums), with the same meaning as in
        :meth:`whoosh.searching.Searcher.search`. If the document is not
        in the filter, the explanation reports `"filtered"` instead of a
        score.
    :param mask: an optional mask, with the same meaning as in
        :meth:`whoosh.searching.Searcher.search`. If the document is in
        the mask, the explanation reports `"masked"` instead of a score.
    :rtype: dict
    """

    reader = searcher.reader()
    if not 0 <= docnum < reader.doc_count_all():
        raise ValueError(f"Document number {docnum} is out of range")

    base = {"docnum": docnum, "query": str(q), "type": type(q).__name__}

    # A deleted document can't produce a meaningful score explanation
    if reader.has_deletions() and reader.is_deleted(docnum):
        return _failure(base, "deleted")

    # A document rejected by a filter/mask can't produce a meaningful score
    # explanation either
    if filter is not None and docnum not in searcher._filter_to_comb(filter):
        return _failure(base, "filtered")
    if mask is not None and docnum in searcher._filter_to_comb(mask):
        return _failure(base, "masked")

    # Find the leaf (per-segment) searcher containing the document and
    # translate the global docnum into the segment-local numbering, the same
    # way the collector does during a real search
    for subsearcher, offset in searcher.leaf_searchers():
        if offset <= docnum < offset + subsearcher.doc_count_all():
            localdoc = docnum - offset
            break
    else:  # pragma: no cover
        raise ValueError(f"Document number {docnum} is out of range")

    # Use the same search context a real search would use, so scores come
    # from the same matchers and the same statistics snapshot
    context = searcher.context()
    weighting = context.weighting or searcher.weighting

    node = _explain_query(q, subsearcher, localdoc, context, weighting)
    node["docnum"] = docnum

    # Apply the weighting model's final() adjustment, if any, the same way
    # the collector does
    if node["matched"] and weighting.use_final:
        node["raw_score"] = node["score"]
        node["score"] = weighting.final(searcher, docnum, node["score"])
        node["details"]["final_adjusted"] = True

    return node


def _failure(node, reason):
    node["matched"] = False
    node["supported"] = True
    node["reason"] = reason
    return node


def _safetext(text):
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8")
        except UnicodeDecodeError:
            return repr(text)
    return text


def _skip_to(matcher, docnum):
    # Advances the matcher to the given document and returns True if the
    # matcher is positioned on the document afterwards
    if not matcher.is_active():
        return False
    if matcher.id() < docnum:
        matcher.skip_to(docnum)
    return matcher.is_active() and matcher.id() == docnum


def _explain_query(q, searcher, docnum, context, weighting):
    node = {
        "type": type(q).__name__,
        "query": str(q),
        "boost": getattr(q, "boost", 1.0),
    }

    try:
        m = q.matcher(searcher, context)
    except TermNotFound:
        m = matching.NullMatcher()

    if not _skip_to(m, docnum):
        node["matched"] = False
        node["supported"] = _is_supported(q)
        if isinstance(q, Phrase):
            node["reason"] = "phrase_no_match"
        elif _is_supported(q):
            node["reason"] = "no_match"
        else:
            node["reason"] = "unsupported_query"
        return node

    node["matched"] = True
    node["score"] = m.score()

    if isinstance(q, Term):
        _explain_term(q, searcher, m, node, weighting)
    elif isinstance(q, Phrase):
        _explain_phrase(q, searcher, docnum, context, weighting, node)
    elif isinstance(q, (And, Or, DisjunctionMax)):
        _explain_compound(q, searcher, docnum, context, weighting, node, m)
    else:
        node["supported"] = False
        node["reason"] = "unsupported_query"

    return node


def _is_supported(q):
    return isinstance(q, (Term, Phrase, And, Or, DisjunctionMax))


def _unwrap(m):
    # Removes boost wrappers to get at the underlying matcher
    while isinstance(m, matching.WrappingMatcher) and not isinstance(
        m, matching.CoordMatcher
    ):
        m = m.child
    return m


def _explain_term(q, searcher, m, node, weighting):
    node["field"] = q.fieldname
    node["term"] = _safetext(q.text)

    field = searcher.schema[q.fieldname]
    text = field.to_bytes(q.text)

    # Give the weighting model the raw postings matcher (without the boost
    # wrapper) so tf is the actual term frequency
    details = weighting.explain(searcher, q.fieldname, text, _unwrap(m))
    if details is None:
        node["supported"] = False
        node["reason"] = "unsupported_weighting_model"
        node["details"] = {"model": type(weighting).__name__}
    else:
        node["supported"] = True
        node["details"] = details


def _explain_phrase(q, searcher, docnum, context, weighting, node):
    node["combine"] = "sum"
    node["details"] = {
        "field": q.fieldname,
        "words": [_safetext(w) for w in q.words],
        "slop": q.slop,
    }

    # A phrase scores as the sum of the scores of its words, so explain each
    # word as a Term child
    children = [
        _explain_query(Term(q.fieldname, word), searcher, docnum, context, weighting)
        for word in q.words
    ]
    node["children"] = children
    node["supported"] = all(child["supported"] for child in children)


def _explain_compound(q, searcher, docnum, context, weighting, node, m):
    details = {}
    if isinstance(q, DisjunctionMax):
        node["combine"] = "max"
        details["tiebreak"] = q.tiebreak
    else:
        node["combine"] = "sum"
        if isinstance(q, Or):
            details["minmatch"] = q.minmatch
            details["scale"] = q.scale
            # If the Or query has a coordination scale, the matcher is
            # wrapped in a CoordMatcher; record its inputs so the score can
            # be recomputed
            cm = m
            if isinstance(cm, matching.WrappingMatcher) and not isinstance(
                cm, matching.CoordMatcher
            ):
                cm = cm.child
            if isinstance(cm, matching.CoordMatcher):
                child = cm.child
                matching_terms = sum(1 for _ in child.matching_terms(child.id()))
                details["coord"] = {
                    "matched_terms": matching_terms,
                    "term_count": cm._termcount,
                }

    node["details"] = details

    children = [
        _explain_query(subq, searcher, docnum, context, weighting)
        for subq in q.subqueries
    ]
    node["children"] = children
    node["supported"] = all(child["supported"] for child in children)
