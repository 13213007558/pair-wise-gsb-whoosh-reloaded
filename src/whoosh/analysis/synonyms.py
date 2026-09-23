"""
Multi-word synonym expansion at query time.

The :class:`SynonymGraphFilter` rewrites a token stream into a token *graph*:
tokens that take part in a synonym expansion are annotated with a ``pos``
(position) and a ``poslen`` (position length) so that every complete, legal
path through the alternatives can be reconstructed later, instead of
crushing all alternatives into a bag of words at the same position.

The :func:`token_paths` function enumerates the paths through such a graph,
up to a configurable limit. The query parser uses it to turn an analyzed
query string into one sub-query per complete path (for example an ``Or`` of
``Phrase`` queries), so mixed paths such as "new city office" (half of one
expansion glued to half of another) can never match.
"""

from bisect import bisect_left

from whoosh.analysis.filters import Filter

#: Default maximum number of expansion paths allowed for a single analyzed
#: query string. See :func:`token_paths`.
DEFAULT_MAX_PATHS = 256


class SynonymPathLimitError(Exception):
    """Raised when enumerating the paths through a synonym token graph would
    produce more paths than the configured limit. The query parser catches
    this and returns a query with its ``error`` attribute set, so the
    condition is detectable instead of silently dropping candidates.
    """


def _word_tuple(obj):
    # Converts a string or sequence of strings into a tuple of words
    if isinstance(obj, str):
        obj = tuple(obj.split())
    else:
        obj = tuple(obj)
    if not obj:
        raise ValueError("Empty word sequence in synonym rule")
    return obj


class SynonymGraphFilter(Filter):
    """A token filter that expands synonyms, including multi-word synonyms,
    into a token graph.

    Rules are supplied by the application as a mapping (or an iterable of
    pairs) from a left-hand side to one or more expansions::

        sf = SynonymGraphFilter({
            "nyc": "new york city",              # one word -> many words
            "new york city": ["nyc"],            # many words -> one word
            "usa": ["united states", "america"], # multiple expansions
        })

    Each left-hand side may be a string (split on whitespace) or a sequence
    of words. Each right-hand side may be a string (a single expansion, split
    on whitespace) or a list of expansions, where each expansion is a string
    or a sequence of words.

    Matching is greedy and deterministic: at each input position the longest
    left-hand side wins, the matched tokens are consumed, and matching
    continues after them. Expansions are never re-scanned, so rules cannot
    trigger each other recursively (no expansion loops).

    If ``keep_orig`` is True (the default), the original token sequence is
    kept as an additional path through the graph alongside its expansions.

    Tokens involved in an expansion are emitted with ``pos`` and ``poslen``
    attributes describing the graph: a token starting at ``pos`` with
    ``poslen`` of N connects to the tokens at position ``pos + N``. Tokens
    following an expansion are shifted so every alternative path rejoins the
    stream at the same position. Tokens are also tagged with ``synregion``
    and ``synalt`` attributes identifying the match and the alternative
    they belong to, so paths cannot mix tokens from different alternatives
    that share a position. Use :func:`token_paths` to enumerate the
    complete token sequences encoded in the output.

    This filter is usually applied at query time only (the index keeps the
    original words, so existing indexes do not need to be rebuilt), for
    example using :class:`whoosh.analysis.filters.MultiFilter`::

        ana = (RegexTokenizer() | LowercaseFilter() | StopFilter()
               | MultiFilter(query=SynonymGraphFilter(synonyms)))
    """

    def __init__(self, synonyms, keep_orig=True):
        """
        :param synonyms: a mapping or iterable of (left, expansions) pairs.
        :param keep_orig: whether to keep the original tokens as an
            alternative path when a rule matches.
        """

        rules = {}
        items = synonyms.items() if hasattr(synonyms, "items") else synonyms
        for lhs, rhs in items:
            lhs = _word_tuple(lhs)
            if isinstance(rhs, str):
                rhs = [rhs]
            expansions = [_word_tuple(alt) for alt in rhs]
            rules.setdefault(lhs, []).extend(expansions)

        self.synonyms = rules
        self.keep_orig = keep_orig
        # Longest left-hand side, for greedy matching
        self._maxlen = max((len(lhs) for lhs in rules), default=0)

    def _match(self, texts, i):
        # Returns (length, expansions) for the longest rule matching the
        # buffered input token texts at index i, or None
        remaining = len(texts) - i
        for length in range(min(self._maxlen, remaining), 0, -1):
            lhs = tuple(texts[i : i + length])
            if lhs in self.synonyms:
                return length, self.synonyms[lhs]
        return None

    def __call__(self, tokens):
        # Buffer the input tokens (copies, since filters must not hold onto
        # the mutable token object) and normalize positions
        buf = []
        for i, t in enumerate(tokens):
            t = t.copy()
            if getattr(t, "pos", None) is None:
                t.pos = i
            buf.append(t)

        out = []
        texts = [t.text for t in buf]
        shift = 0
        region = 0
        i = 0
        while i < len(buf):
            match = self._match(texts, i)
            if match is None:
                t = buf[i]
                t.pos += shift
                out.append(t)
                i += 1
                continue

            length, expansions = match
            matched = buf[i : i + length]
            base = matched[0].pos + shift
            startchar = getattr(matched[0], "startchar", None)
            endchar = getattr(matched[-1], "endchar", None)

            # The alternatives for this region: the expansions plus,
            # optionally, the original token sequence
            alternatives = list(expansions)
            original = tuple(t.text for t in matched)
            if self.keep_orig and original not in alternatives:
                alternatives.append(original)

            # The region spans the length of the longest alternative, so
            # every path through the region ends at base + span
            span = max(len(alt) for alt in alternatives)
            for altnum, alt in enumerate(alternatives):
                lastlen = span - len(alt) + 1
                for j, word in enumerate(alt):
                    if alt == original and j < length:
                        # Keep the attributes of the original token
                        t = matched[j].copy()
                    else:
                        t = matched[min(j, length - 1)].copy()
                        t.text = word
                        t.stopped = False
                        if startchar is not None:
                            t.startchar = startchar
                        if endchar is not None:
                            t.endchar = endchar
                    t.pos = base + j
                    t.poslen = lastlen if j == len(alt) - 1 else 1
                    # Scope the token to its alternative within this match,
                    # so paths cannot mix tokens from different
                    # alternatives that happen to share a position
                    t.synregion = region
                    t.synalt = altnum
                    out.append(t)

            region += 1
            shift += span - length
            i += length

        return out


def is_graph_stream(tokens):
    """Returns True if any token in the list carries synonym graph
    information (a ``poslen`` attribute), as emitted by
    :class:`SynonymGraphFilter`.
    """

    return any(hasattr(t, "poslen") for t in tokens)


def token_paths(tokens, max_paths=DEFAULT_MAX_PATHS):
    """Enumerates the complete token paths through a token graph, as
    emitted by :class:`SynonymGraphFilter`.

    A token at position P with a ``poslen`` of N (default 1) connects to
    the tokens at the next available position greater than or equal to
    P + N. (Positions may have holes, for example where stop words were
    removed without renumbering; a path simply skips over the hole, and
    :func:`path_position_gap` reports the total size of the gaps so the
    query can compensate with slop.)

    :param tokens: a sequence of Token objects with ``pos`` attributes.
    :param max_paths: the maximum number of paths allowed. If the graph
        would produce more paths than this, :class:`SynonymPathLimitError`
        is raised instead of silently truncating the results.
    :returns: a list of paths, where each path is a list of Token objects.
    """

    tokens = list(tokens)
    if not tokens:
        return []

    by_pos = {}
    for t in tokens:
        by_pos.setdefault(t.pos, []).append(t)
    positions = sorted(by_pos)

    def next_pos(after):
        i = bisect_left(positions, after)
        return positions[i] if i < len(positions) else None

    def compatible(a, b):
        # Tokens produced by the same synonym match may only chain to
        # tokens from the same alternative, so a path can never mix
        # different alternatives that share a position
        aregion = getattr(a, "synregion", None)
        bregion = getattr(b, "synregion", None)
        if aregion is None or bregion is None or aregion != bregion:
            return True
        return getattr(a, "synalt", None) == getattr(b, "synalt", None)

    def successors(t):
        np = next_pos(t.pos + getattr(t, "poslen", 1))
        if np is None:
            return []
        return [u for u in by_pos[np] if compatible(t, u)]

    # Sources are tokens that no other token connects to
    has_incoming = set()
    for t in tokens:
        for u in successors(t):
            has_incoming.add(id(u))
    sources = [t for t in tokens if id(t) not in has_incoming]

    paths = []

    def walk(t, path):
        path = path + [t]
        nexts = successors(t)
        if not nexts:
            paths.append(path)
            if len(paths) > max_paths:
                raise SynonymPathLimitError(
                    "Synonym expansion produced more than %d token paths; "
                    "refine the query or raise the path limit" % max_paths
                )
            return
        for u in nexts:
            walk(u, path)

    for t in sources:
        walk(t, [])

    return paths


def path_position_gap(path):
    """Returns the total size of the position holes along a token path
    (for example from removed stop words that were not renumbered). A
    phrase query built from the path should add this to its slop to
    preserve the meaning of the gaps.
    """

    extra = 0
    for a, b in zip(path, path[1:]):
        gap = b.pos - (a.pos + getattr(a, "poslen", 1))
        if gap > 0:
            extra += gap
    return extra
