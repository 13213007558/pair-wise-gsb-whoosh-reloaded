# Copyright 2007 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
# THIS SOFTWARE IS PROVIDED BY MATT CHAPUT ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MATT CHAPUT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

"""Multi-word synonym expansion for query-time analysis.

This module implements a :class:`SynonymGraphFilter` analysis filter that
applies application-provided, multi-word synonym rules (for example
``nyc` -> ``new york city`) to a token stream, producing a token *graph*
instead of a flat token list. Tokens in the graph carry a `poslength`
attribute (default 1) describing how many positions they span, and a
`hole_before` attribute recording positions skipped by removed stop words.

The query parser understands these graph tokens and turns them into one
query per complete, legal token path through the graph (combined with
:class:`whoosh.query.Or`), instead of squashing all alternatives into a
single position. This means a search for ``visit nyc office` can match
``visit new york city office`, but can never match a mixed path such as
``new city office`.

Because expansion happens at query time, the index keeps the original words;
existing indexes do not need to be rebuilt.
"""

from whoosh.analysis.acore import Token
from whoosh.analysis.filters import Filter

#: Default cap on the number of token paths a single token graph may expand
#: into. Exceeding the cap raises :class:`SynonymExpansionError` instead of
#: silently dropping candidate paths.
DEFAULT_MAX_PATHS = 256


class SynonymExpansionError(Exception):
    """Raised when a synonym token graph would expand into more token paths
    than the configured limit. This is a distinct exception type so callers
    can tell expansion-limit failures apart from other errors; candidates are
    never silently truncated.
    """

    pass


def _wordtuple(words):
    if isinstance(words, str):
        words = words.split()
    return tuple(words)


class SynonymMap:
    """A set of multi-word synonym rules supplied by the application.

    Each rule maps a sequence of one or more words (the left-hand side) to a
    replacement sequence of one or more words (the right-hand side). Rules
    support word-to-multi-word (``nyc` -> ``new york city`),
    multi-word-to-word (``new york city` -> ``nyc`), and optionally
    keeping the original words as an additional alternative
    (``keep_orig=True`).

    Matching is deterministic: at each token position the rule with the
    longest left-hand side wins. When several rules share the same left-hand
    side, they are applied as alternatives in the fixed order they were
    added. Tokens consumed by a matched rule are not considered again, so
    overlapping rules are always resolved the same way. Expansions produced
    by a rule are never re-scanned for further matches, so chains or cycles
    of rules cannot cause recursive expansion.
    """

    def __init__(self):
        # Maps lhs tuple -> list of (rhs tuple, keep_orig) in insertion order
        self._rules = {}
        self._maxlen = 0

    def __eq__(self, other):
        return (
            other and self.__class__ is other.__class__ and self._rules == other._rules
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self._rules!r})"

    def __len__(self):
        return sum(len(rules) for rules in self._rules.values())

    def add(self, lhs, rhs, keep_orig=False):
        """Adds a synonym rule.

        :param lhs: the word sequence to match, either a string of
            space-separated words or a sequence of word strings.
        :param rhs: the replacement word sequence, either a string of
            space-separated words or a sequence of word strings.
        :param keep_orig: if True, the original words are kept as an
            additional alternative alongside the replacement.
        """

        lhs = _wordtuple(lhs)
        rhs = _wordtuple(rhs)
        if not lhs or not rhs:
            raise ValueError("Synonym rules need non-empty word sequences")
        self._rules.setdefault(lhs, []).append((rhs, bool(keep_orig)))
        self._maxlen = max(self._maxlen, len(lhs))

    def match(self, texts, pos):
        """Returns the rules matching the word sequence `texts` starting at
        index `pos`, or None if no rule matches.

        Returns a pair `(length, rules)` where `length` is the number of
        words consumed by the longest matching left-hand side and `rules`
        is the list of `(rhs, keep_orig)` pairs for that left-hand side, in
        the fixed order the rules were added.
        """

        maxlen = min(self._maxlen, len(texts) - pos)
        for length in range(maxlen, 0, -1):
            key = tuple(texts[pos : pos + length])
            rules = self._rules.get(key)
            if rules:
                return length, rules
        return None


def has_synonym_filter(analyzer):
    """Returns True if the given analyzer (or composed analyzer chain)
    contains a :class:`SynonymGraphFilter`.
    """

    items = getattr(analyzer, "items", None)
    if items is None:
        return isinstance(analyzer, SynonymGraphFilter)
    return any(isinstance(item, SynonymGraphFilter) for item in items)


class SynonymGraphFilter(Filter):
    """A query-time analysis filter that expands multi-word synonyms into a
    token graph.

    The filter consumes the (already tokenized) query token stream and emits
    tokens annotated with:

    ``pos`
        The token's start position in the expanded graph.
    ``poslength`
        How many positions the token spans (always 1 for ordinary tokens).
    ``hole_before`
        The number of positions skipped before this token in the original
        stream (for example positions left empty by removed stop words).

    Consumers such as the query parser use these attributes to enumerate the
    complete legal token paths through the graph (see :func:`graph_paths`).

    By default the filter only expands tokens in ``query` mode; in any
    other mode (for example while indexing) it passes the stream through
    unchanged, so indexed positions are not affected and existing indexes
    keep working.

    >>> from whoosh import analysis
    >>> synmap = analysis.SynonymMap()
    >>> synmap.add("nyc", "new york city", keep_orig=True)
    >>> ana = analysis.RegexTokenizer() | analysis.SynonymGraphFilter(synmap)
    >>> [(t.text, t.pos, t.poslength) for t in ana("visit nyc office",
    ...     mode="query", positions=True)]
    [('visit', 0, 1), ('nyc', 1, 3), ('new', 1, 1), ('york', 2, 1), ('city', 3, 1), ('office', 4, 1)]
    """

    def __init__(self, synmap, max_paths=DEFAULT_MAX_PATHS, query_only=True):
        """
        :param synmap: a :class:`SynonymMap` containing the rules to apply.
        :param max_paths: the maximum number of token paths the expanded
            graph may contain. If expansion would produce more paths,
            :class:`SynonymExpansionError` is raised instead of silently
            dropping candidates.
        :param query_only: if True (the default), only expand when the token
            stream is being analyzed in ``query` mode; pass tokens through
            unchanged otherwise (e.g. at indexing time).
        """

        self.synmap = synmap
        self.max_paths = max_paths
        self.query_only = query_only

    def __call__(self, tokens):
        # Materialize the stream: we need random access for multi-word
        # matching, and we must not hold on to the mutable Token objects
        toks = [t.copy() for t in tokens]
        if not toks:
            return

        if self.query_only and getattr(toks[0], "mode", "") != "query":
            # Not analyzing a query (e.g. indexing): pass the original
            # tokens through untouched
            for t in toks:
                yield t
            return

        if getattr(toks[0], "tokenize", True) is False:
            # The caller asked for the text not to be tokenized/transformed
            # (e.g. wildcard or prefix queries): pass tokens through
            for t in toks:
                yield t
            return

        count = len(toks)

        # Make sure every token has a position
        for i, t in enumerate(toks):
            if not hasattr(t, "pos"):
                t.pos = i

        # Record position holes (e.g. from removed stop words) in the
        # original stream
        holes = [0] * count
        for i in range(1, count):
            holes[i] = max(0, toks[i].pos - toks[i - 1].pos - 1)

        texts = [t.text for t in toks]

        # Split the stream into regions. Each region is a tuple
        # (first_token, last_token, alternatives) covering the consumed
        # original tokens. Each alternative is a pair (specs, chars) where
        # specs is a list of (source_token, text, relpos, hole_before) and
        # chars is None for the original tokens or the (first, last) consumed
        # tokens for an expansion (used to compute character ranges).
        regions = []
        i = 0
        while i < count:
            t = toks[i]
            matched = None if t.stopped else self.synmap.match(texts, i)
            if matched is None:
                regions.append((t, t, [([(t, t.text, 0, holes[i])], None)]))
                i += 1
                continue

            length, rules = matched
            consumed = toks[i : i + length]
            first, last = consumed[0], consumed[-1]
            alternatives = []
            keep = False
            for rhs, keep_orig in rules:
                specs = []
                for j, word in enumerate(rhs):
                    specs.append((first, word, j, holes[i] if j == 0 else 0))
                alternatives.append((specs, (first, last)))
                keep = keep or keep_orig
            if keep:
                specs = []
                for k, tok in enumerate(consumed):
                    specs.append((tok, tok.text, tok.pos - first.pos, holes[i + k]))
                alternatives.insert(0, (specs, None))
            regions.append((first, last, alternatives))
            i += length

        # Assign graph positions region by region and emit tokens
        shift = 0
        path_count = 1
        for first, last, alternatives in regions:
            path_count *= len(alternatives)
            if path_count > self.max_paths:
                raise SynonymExpansionError(
                    "Synonym expansion of %r exceeds the limit of %d token "
                    "paths" % (" ".join(texts), self.max_paths)
                )

            base = first.pos + shift
            # End position of the original tokens in shifted space
            orig_end = base + (last.pos - first.pos) + 1
            # All alternatives in a region must end at the same position so
            # that paths cannot mix tokens from different alternatives
            region_end = orig_end
            for specs, chars in alternatives:
                if chars is not None:
                    region_end = max(region_end, base + len(specs))

            for specs, chars in alternatives:
                lastindex = len(specs) - 1
                for j, (src, text, relpos, hole) in enumerate(specs):
                    t = src.copy()
                    t.text = text
                    t.pos = base + relpos
                    t.poslength = (region_end - t.pos) if j == lastindex else 1
                    t.hole_before = hole
                    t.positions = True
                    if chars is not None:
                        # Expansion tokens map back to the character range of
                        # the original text they replace (for highlighting)
                        first_c, last_c = chars
                        t.startchar = getattr(first_c, "startchar", None)
                        t.endchar = getattr(last_c, "endchar", None)
                    yield t

            shift += region_end - orig_end


def graph_paths(tokens, max_paths=DEFAULT_MAX_PATHS):
    """Enumerates the complete legal token paths through a token graph
    produced by :class:`SynonymGraphFilter`.

    A legal path starts at the graph's first position, ends at its last
    position, and only chains tokens whose positions connect (a token may be
    followed by any token starting at the smallest position at or after the
    token's end). Because every alternative in a synonym region ends at the
    same position, paths can never mix tokens from different alternatives
    (so ``new city office` is never a path).

    :param tokens: a sequence of token objects with `pos`, `poslength`,
        and `hole_before` attributes.
    :param max_paths: the maximum number of paths to enumerate. If the graph
        contains more paths, :class:`SynonymExpansionError` is raised
        instead of silently truncating the result.
    :returns: a list of paths, where each path is a list of tokens.
    """

    tokens = list(tokens)
    if not tokens:
        return []

    for t in tokens:
        if not hasattr(t, "poslength"):
            t.poslength = 1
        if not hasattr(t, "hole_before"):
            t.hole_before = 0

    by_pos = {}
    for t in tokens:
        by_pos.setdefault(t.pos, []).append(t)
    positions = sorted(by_pos)
    start = positions[0]
    end = max(t.pos + t.poslength for t in tokens)

    paths = []

    def walk(token, path):
        endpos = token.pos + token.poslength
        path = path + [token]
        if endpos >= end:
            paths.append(path)
            if len(paths) > max_paths:
                raise SynonymExpansionError(
                    "Token graph expands to more than %d paths" % max_paths
                )
            return
        nextpos = None
        for p in positions:
            if p >= endpos:
                nextpos = p
                break
        if nextpos is None:
            return
        for follower in by_pos[nextpos]:
            walk(follower, path)

    for token in by_pos[start]:
        walk(token, [])

    return paths
