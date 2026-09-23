# Copyright 2007 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
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
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

"""
This module contains helper functions for correcting typos in user queries.
"""

import math

from bisect import bisect_left
from heapq import heappush, heapreplace

from whoosh import highlight

# Corrector objects


class Corrector:
    """
    Base class for spelling correction objects. Concrete sub-classes should
    implement the ``_suggestions`` method.
    """

    def suggest(self, text, limit=5, maxdist=2, prefix=0):
        """
        :param text: the text to check. This word will **not** be added to the
            suggestions, even if it appears in the word graph.
        :param limit: only return up to this many suggestions. If there are not
            enough terms in the field within ``maxdist`` of the given word, the
            returned list will be shorter than this number.
        :param maxdist: the largest edit distance from the given word to look
            at. Values higher than 2 are not very effective or efficient.
        :param prefix: require suggestions to share a prefix of this length
            with the given word. This is often justifiable since most
            misspellings do not involve the first letter of the word. Using a
            prefix dramatically decreases the time it takes to generate the
            list of words.
        """

        _suggestions = self._suggestions

        heap = []
        for item in _suggestions(text, maxdist, prefix):
            # Note that the *higher* scores (item[0]) are better!
            if len(heap) < limit:
                heappush(heap, item)
            elif item > heap[0]:
                heapreplace(heap, item)

        sugs = sorted(heap, key=lambda x: (0 - x[0], x[1]))
        return [sug for _, sug in sugs]

    def _suggestions(self, text, maxdist, prefix):
        """
        Low-level method that yields a series of (score, "suggestion")
        tuples.

        :param text: the text to check.
        :param maxdist: the maximum edit distance.
        :param prefix: require suggestions to share a prefix of this length
            with the given word.
        """

        raise NotImplementedError


class ReaderCorrector(Corrector):
    """
    Suggests corrections based on the content of a field in a reader.

    Ranks suggestions by the edit distance, then by highest to lowest
    frequency.
    """

    def __init__(self, reader, fieldname, fieldobj):
        self.reader = reader
        self.fieldname = fieldname
        self.fieldobj = fieldobj

    def _suggestions(self, text, maxdist, prefix):
        reader = self.reader
        freq = reader.frequency

        fieldname = self.fieldname
        fieldobj = reader.schema[fieldname]
        sugfield = fieldobj.spelling_fieldname(fieldname)

        for sug in reader.terms_within(sugfield, text, maxdist, prefix=prefix):
            # Higher scores are better, so negate the distance and frequency
            f = freq(fieldname, sug) or 1
            score = 0 - (maxdist + (1.0 / f * 0.5))
            yield (score, sug)


class ListCorrector(Corrector):
    """
    Suggests corrections based on the content of a sorted list of strings.
    """

    def __init__(self, wordlist):
        self.wordlist = wordlist

    def _suggestions(self, text, maxdist, prefix):
        from whoosh.automata.fsa import find_all_matches
        from whoosh.automata.lev import levenshtein_automaton

        seen = set()
        for mxd in range(1, maxdist + 1):
            dfa = levenshtein_automaton(text, mxd, prefix).to_dfa()
            sk = self.Skipper(self.wordlist)
            for sug in find_all_matches(dfa, sk):
                if sug not in seen:
                    seen.add(sug)
                    yield (0 - mxd), sug

    class Skipper:
        def __init__(self, data):
            self.data = data
            self.i = 0

        def __call__(self, w):
            if self.data[self.i] == w:
                return w
            self.i += 1
            pos = bisect_left(self.data, w, self.i)
            if pos < len(self.data):
                return self.data[pos]
            else:
                return None


class MultiCorrector(Corrector):
    """
    Merges suggestions from a list of sub-correctors.
    """

    def __init__(self, correctors, op):
        self.correctors = correctors
        self.op = op

    def _suggestions(self, text, maxdist, prefix):
        op = self.op
        seen = {}
        for corr in self.correctors:
            for score, sug in corr._suggestions(text, maxdist, prefix):
                if sug in seen:
                    seen[sug] = op(seen[sug], score)
                else:
                    seen[sug] = score
        return seen.items()


# Query correction


class Correction:
    """
    Represents the corrected version of a user query string. Has the
    following attributes:

    ``query``
        The corrected :class:`whoosh.query.Query` object.
    ``string``
        The corrected user query string.
    ``original_query``
        The original :class:`whoosh.query.Query` object that was corrected.
    ``original_string``
        The original user query string.
    ``tokens``
        A list of token objects representing the corrected words.

    You can also use the :meth:`Correction.format_string` method to reformat the
    corrected query string using a :class:`whoosh.highlight.Formatter` class.
    For example, to display the corrected query string as HTML with the
    changed words emphasized::

        from whoosh import highlight

        correction = mysearcher.correct_query(q, qstring)

        hf = highlight.HtmlFormatter(classname="change")
        html = correction.format_string(hf)
    """

    def __init__(self, q, qstring, corr_q, tokens):
        self.original_query = q
        self.query = corr_q
        self.original_string = qstring
        self.tokens = tokens

        if self.original_string:
            self.string = self.format_string(highlight.NullFormatter())
        else:
            self.string = ""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.query!r}, {self.string!r})"

    def format_string(self, formatter):
        """
        Highlights the corrected words in the original query string using the
        given :class:`~whoosh.highlight.Formatter`.

        :param formatter: A :class:`whoosh.highlight.Formatter` instance.
        :return: the output of the formatter (usually a string).
        """

        if not self.original_string:
            return ""
        if isinstance(formatter, type):
            formatter = formatter()

        fragment = highlight.Fragment(self.original_string, self.tokens)
        out = formatter.format_fragment(fragment, replace=True)
        return out


# QueryCorrector objects


class QueryCorrector:
    """
    Base class for objects that correct words in a user query.
    """

    def __init__(self, fieldname):
        self.fieldname = fieldname

    def correct_query(self, q, qstring):
        """
        Returns a :class:`Correction` object representing the corrected
        form of the given query.

        :param q: the original :class:`whoosh.query.Query` tree to be
            corrected.
        :param qstring: the original user query. This may be None if the
            original query string is not available, in which case the
            ``Correction.string`` attribute will also be None.
        :rtype: :class:`Correction`
        """

        raise NotImplementedError

    def field(self):
        return self.fieldname


class SimpleQueryCorrector(QueryCorrector):
    """
    A simple query corrector based on a mapping of field names to
    :class:`Corrector` objects, and a list of ``("fieldname", "text")`` tuples
    to correct. And terms in the query that appear in list of term tuples are
    corrected using the appropriate corrector.
    """

    def __init__(self, correctors, terms, aliases=None, prefix=0, maxdist=2):
        """
        :param correctors: a dictionary mapping field names to
            :class:`Corrector` objects.
        :param terms: a sequence of ``("fieldname", "text")`` tuples
            representing terms to be corrected.
        :param aliases: a dictionary mapping field names in the query to
            field names for spelling suggestions.
        :param prefix: suggested replacement words must share this number of
            initial characters with the original word. Increasing this even to
            just ``1`` can dramatically speed up suggestions, and may be
            justifiable since spellling mistakes rarely involve the first
            letter of a word.
        :param maxdist: the maximum number of "edits" (insertions, deletions,
            subsitutions, or transpositions of letters) allowed between the
            original word and any suggestion. Values higher than ``2`` may be
            slow.
        """

        self.correctors = correctors
        self.aliases = aliases or {}
        self.termset = frozenset(terms)
        self.prefix = prefix
        self.maxdist = maxdist

    def correct_query(self, q, qstring):
        correctors = self.correctors
        aliases = self.aliases
        termset = self.termset
        prefix = self.prefix
        maxdist = self.maxdist

        # A list of tokens that were changed by a corrector
        corrected_tokens = []

        # The corrected query tree. We don't need to deepcopy the original
        # because we use Query.replace() to find-and-replace the corrected
        # words and it returns a copy of the query tree.
        corrected_q = q

        # For every word in the original query...
        # Note we can't put these in a set, because we must preserve WHERE
        # in the query each token occured so we can format them later
        for token in q.all_tokens():
            fname = token.fieldname
            aname = aliases.get(fname, fname)

            # If this is one of the words we're supposed to correct...
            if (fname, token.text) in termset:
                c = correctors[aname]
                sugs = c.suggest(token.text, prefix=prefix, maxdist=maxdist)
                if sugs:
                    # This is a "simple" corrector, so we just pick the first
                    # suggestion :/
                    sug = sugs[0]

                    # Return a new copy of the original query with this word
                    # replaced by the correction
                    corrected_q = corrected_q.replace(token.fieldname, token.text, sug)
                    # Add the token to the list of corrected tokens (for the
                    # formatter to use later)
                    token.original = token.text
                    token.text = sug
                    corrected_tokens.append(token)

        return Correction(q, qstring, corrected_q, corrected_tokens)


# Contextual (split/merge) query correction


class CorrectionCosts:
    """
    Configurable costs for the edit operations used by
    ContextualQueryCorrector.

    "insert", "delete" and "replace" are the per-character costs used when
    computing the weighted edit distance between a query word and a
    dictionary word. "split" is the extra cost of splitting one query word
    into two dictionary words, and "merge" is the extra cost of merging two
    adjacent query words into one dictionary word (both are added to any
    per-character edit costs of the parts).
    """

    def __init__(self, insert=1.0, delete=1.0, replace=1.0, split=1.0,
                 merge=1.0):
        self.insert = insert
        self.delete = delete
        self.replace = replace
        self.split = split
        self.merge = merge

    def __repr__(self):
        return (
            "%s(insert=%r, delete=%r, replace=%r, split=%r, merge=%r)"
            % (self.__class__.__name__, self.insert, self.delete,
               self.replace, self.split, self.merge)
        )


def _edit_cost(a, b, costs):
    # Weighted edit distance between strings a and b using the per-character
    # insert/delete/replace costs from a CorrectionCosts object.
    la, lb = len(a), len(b)
    if not la:
        return lb * costs.insert
    if not lb:
        return la * costs.delete
    prev = [j * costs.insert for j in range(lb + 1)]
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i * costs.delete] + [0.0] * lb
        for j in range(1, lb + 1):
            sub = prev[j - 1] if ca == b[j - 1] else prev[j - 1] + costs.replace
            cur[j] = min(sub, prev[j] + costs.delete, cur[j - 1] + costs.insert)
        prev = cur
    return prev[lb]


class QueryChange:
    """
    Represents a single replacement in a corrected query string.

    "start" and "end" are character offsets into the *original* query
    string, "original" is the text that was replaced, "replacement" is the
    new text, and "reason" is a human-readable explanation of why the
    change was made.
    """

    def __init__(self, start, end, original, replacement, reason):
        self.start = start
        self.end = end
        self.original = original
        self.replacement = replacement
        self.reason = reason

    def __repr__(self):
        return "%s(%r:%r, %r -> %r)" % (
            self.__class__.__name__, self.start, self.end,
            self.original, self.replacement,
        )

    def __eq__(self, other):
        return (
            isinstance(other, QueryChange)
            and (self.start, self.end, self.original, self.replacement,
                 self.reason)
            == (other.start, other.end, other.original, other.replacement,
                other.reason)
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class RewrittenQuery:
    """
    One possible rewrite of a query string, as produced by
    :meth:ContextualQueryCorrector.correct.

    "text" is the rewritten query string, "cost" is the total edit cost,
    "context_score" is the accumulated context (unigram/bigram) score,
    "score" is the final ranking value (lower is better), and "changes" is
    a list of :class:QueryChange objects with offsets into the original
    query string.
    """

    def __init__(self, text, cost, context_score, score, changes):
        self.text = text
        self.cost = cost
        self.context_score = context_score
        self.score = score
        self.changes = changes

    def __repr__(self):
        return "%s(%r, score=%.3f)" % (
            self.__class__.__name__, self.text, self.score,
        )


class QueryCorrection:
    """
    The result of correcting a raw query string with
    :class:ContextualQueryCorrector.

    "original" is the original query string. "text" and "changes" are the
    best rewrite and its list of :class:QueryChange objects. "alternatives"
    is the ranked list of the top-K :class:RewrittenQuery objects (the
    first one is the best). "exhaustive" is False when the candidate or
    search-state limits were hit, meaning cheaper rewrites may exist that
    were not explored.
    """

    def __init__(self, original, alternatives, exhaustive):
        self.original = original
        self.alternatives = alternatives
        self.exhaustive = exhaustive
        if alternatives:
            best = alternatives[0]
            self.text = best.text
            self.changes = best.changes
        else:
            self.text = original
            self.changes = []

    def __repr__(self):
        return "%s(%r -> %r, exhaustive=%r)" % (
            self.__class__.__name__, self.original, self.text,
            self.exhaustive,
        )


# Query string scanning

# Tokens containing any of these characters are considered query syntax and
# are never corrected.
_SYNTAX_CHARS = frozenset(":*?^~")
# Characters that delimit tokens and are always treated as syntax.
_BRACKET_CHARS = frozenset("()[]{}\"")
# Boolean operator tokens are never corrected.
_BOOL_TOKENS = frozenset(("AND", "OR", "NOT", "ANDNOT", "&&", "||", "!"))


def _scan_fragments(qstring):
    # Splits a query string into runs of correctable plain words. Returns a
    # list of fragments, where each fragment is a list of (start, end, word)
    # tuples with character offsets into qstring. Quoted spans, field
    # prefixes, boolean operators, wildcards, boosts, groups and +/- clauses
    # are treated as syntax and left out of the fragments.
    fragments = []
    current = []
    in_quote = False
    prev_field_prefix = False
    i, n = 0, len(qstring)

    def close_fragment():
        if current:
            fragments.append(current[:])
            del current[:]

    while i < n:
        ch = qstring[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _BRACKET_CHARS:
            if ch == "\"":
                in_quote = not in_quote
            close_fragment()
            prev_field_prefix = False
            i += 1
            continue
        start = i
        while i < n and not qstring[i].isspace() \
                and qstring[i] not in _BRACKET_CHARS:
            i += 1
        token = qstring[start:i]
        protected = (
            in_quote
            or prev_field_prefix
            or token in _BOOL_TOKENS
            or token[0] in "+-"
            or any(c in token for c in _SYNTAX_CHARS)
            or token.endswith(":")
        )
        if protected:
            close_fragment()
        else:
            current.append((start, i, token))
        prev_field_prefix = token.endswith(":")
    close_fragment()
    return fragments


def _apply_changes(qstring, changes):
    # Splices the given changes (with offsets into qstring) into the query
    # string and returns the rewritten string.
    out = []
    pos = 0
    for ch in sorted(changes, key=lambda c: (c.start, c.end)):
        out.append(qstring[pos:ch.start])
        out.append(ch.replacement)
        pos = ch.end
    out.append(qstring[pos:])
    return "".join(out)


class _FragState:
    # One search state while correcting a fragment of plain words.
    __slots__ = ("cost", "context", "out", "changes")

    def __init__(self, cost, context, out, changes):
        self.cost = cost
        self.context = context
        self.out = out
        self.changes = changes


class ContextualQueryCorrector:
    """
    Corrects raw user query strings using a dictionary (read from an index
    reader or a supplied word list) and application-supplied adjacent-word
    (bigram) frequencies.

    Unlike :class:SimpleQueryCorrector, this corrector works on whole runs
    of plain words at once: besides per-word insert/delete/replace edits it
    can split one word into two dictionary words (newyork -> new york) and
    merge two adjacent words into one (hot el -> hotel), and it ranks the
    rewrites by total edit cost combined with a context score derived from
    word and bigram frequencies. No online model is used; the index is only
    ever read from.

    Query syntax is preserved: quoted phrases, field: prefixes, boolean
    operators, wildcards, boosts and groups are copied to the output
    verbatim and only runs of plain words are eligible for correction.
    """

    def __init__(self, reader=None, fieldname=None, wordlist=None,
                 word_freq=None, bigram_freq=None, costs=None,
                 context_weight=0.5, oov_cost=1.0, allow_split=True,
                 allow_merge=True, max_op_cost=2.0, max_candidates=32,
                 max_states=64, top_k=5):
        """
        :param reader: an index reader used as the source of the dictionary
            and unigram frequencies. The reader is only read from.
        :param fieldname: the field to read the dictionary from (required
            when reader is given).
        :param wordlist: an explicit list of dictionary words, used instead
            of a reader.
        :param word_freq: optional mapping of word -> frequency, used with
            wordlist (defaults to 1 for every word).
        :param bigram_freq: optional mapping of (word1, word2) -> frequency
            supplied by the application, used for the context score.
        :param costs: a CorrectionCosts object with the per-operation costs.
        :param context_weight: how strongly the context score influences
            ranking; the final score is cost - context_weight * context.
        :param oov_cost: the cost of keeping a word that is not in the
            dictionary (words in the dictionary are kept for free).
        :param allow_split: allow splitting one word into two.
        :param allow_merge: allow merging two adjacent words into one.
        :param max_op_cost: the maximum total cost of a single
            replace/split/merge operation.
        :param max_candidates: the maximum number of replacement candidates
            considered per word (and per split/merge operation).
        :param max_states: the maximum number of search states kept per
            word position (beam width).
        :param top_k: the maximum number of alternatives returned.
        """

        if reader is None and wordlist is None:
            raise ValueError("Either reader or wordlist is required")
        if reader is not None:
            if fieldname is None:
                raise ValueError("fieldname is required with a reader")
            fieldobj = reader.schema[fieldname]
            sugfield = fieldobj.spelling_fieldname(fieldname)
            words = []
            freqs = {}
            for bterm in reader.lexicon(sugfield):
                word = fieldobj.from_bytes(bterm)
                words.append(word)
                freqs[word] = reader.frequency(fieldname, word) or 1
        else:
            words = sorted(wordlist)
            word_freq = word_freq or {}
            freqs = dict((w, word_freq.get(w, 1)) for w in words)

        self.words = words
        self.word_set = frozenset(words)
        self.freqs = freqs
        self.bigram_freq = dict(bigram_freq or {})
        self.costs = costs or CorrectionCosts()
        self.context_weight = context_weight
        self.oov_cost = oov_cost
        self.allow_split = allow_split
        self.allow_merge = allow_merge
        self.max_op_cost = max_op_cost
        self.max_candidates = max_candidates
        self.max_states = max_states
        self.top_k = top_k
        # Set to True whenever a limit truncates the search.
        self._limit_hit = False

    # Candidate generation

    def _word_candidates(self, word):
        # Returns (candidates, truncated) where candidates is a sorted list
        # of (edit_cost, word) tuples within max_op_cost of the given word,
        # excluding the word itself.
        results = []
        maxcost = self.max_op_cost
        for w in self.words:
            if w == word:
                continue
            c = _edit_cost(word, w, self.costs)
            if c <= maxcost:
                results.append((c, w))
        results.sort(key=lambda t: (t[0], -self.freqs.get(t[1], 0), t[1]))
        truncated = False
        if len(results) > self.max_candidates:
            results = results[: self.max_candidates]
            truncated = True
        return results, truncated

    def word_candidates(self, word):
        """
        Returns the per-word replacement candidates for the given word as a
        list of (edit_cost, word) tuples, mirroring the behavior of the
        classic per-word correctors when split/merge are disabled.
        """

        return self._word_candidates(word)[0]

    def _cands_incl_exact(self, word):
        cands, truncated = self._word_candidates(word)
        if word in self.word_set:
            cands = [(0.0, word)] + cands
        return cands, truncated

    def _split_candidates(self, word):
        # Returns (candidates, truncated) where candidates is a sorted list
        # of (cost, (word1, word2)) tuples for splitting the word into two
        # dictionary words.
        best = {}
        truncated = False
        for i in range(1, len(word)):
            lefts, trunc = self._cands_incl_exact(word[:i])
            truncated = truncated or trunc
            rights, trunc = self._cands_incl_exact(word[i:])
            truncated = truncated or trunc
            for cl, wl in lefts:
                if self.costs.split + cl > self.max_op_cost:
                    continue
                for cr, wr in rights:
                    total = self.costs.split + cl + cr
                    if total > self.max_op_cost:
                        continue
                    pair = (wl, wr)
                    if pair not in best or total < best[pair]:
                        best[pair] = total
        results = sorted(
            ((c, pair) for pair, c in best.items()),
            key=lambda t: (t[0], t[1]),
        )
        if len(results) > self.max_candidates:
            results = results[: self.max_candidates]
            truncated = True
        return results, truncated

    def _merge_candidates(self, word1, word2):
        # Returns (candidates, truncated) where candidates is a sorted list
        # of (cost, word) tuples for merging two adjacent query words into
        # one dictionary word.
        cands, truncated = self._cands_incl_exact(word1 + word2)
        results = []
        for c, w in cands:
            total = self.costs.merge + c
            if total <= self.max_op_cost:
                results.append((total, w))
        results.sort(key=lambda t: (t[0], t[1]))
        if len(results) > self.max_candidates:
            results = results[: self.max_candidates]
            truncated = True
        return results, truncated

    # Fragment correction

    def _context_bonus(self, prev_word, word):
        bonus = math.log1p(self.freqs.get(word, 0))
        if prev_word is not None:
            bonus += math.log1p(self.bigram_freq.get((prev_word, word), 0))
        return bonus

    def _rank_key(self, state):
        return (
            state.cost - self.context_weight * state.context,
            state.out,
        )

    def _advance(self, buckets, pos, state, op_cost, new_words, change):
        context = state.context
        prev = state.out[-1] if state.out else None
        for w in new_words:
            context += self._context_bonus(prev, w)
            prev = w
        changes = state.changes
        if change is not None:
            changes = changes + (change,)
        buckets.setdefault(pos, []).append(
            _FragState(state.cost + op_cost, context,
                       state.out + tuple(new_words), changes)
        )

    def _prune(self, bucket):
        if len(bucket) > self.max_states:
            bucket.sort(key=self._rank_key)
            del bucket[self.max_states:]
            self._limit_hit = True

    def correct_fragment(self, words, spans=None):
        """
        Corrects a run of plain words. Returns a tuple (alternatives,
        exhaustive) where alternatives is a ranked list of RewrittenQuery
        objects whose changes have offsets relative to the given spans (or
        relative to a synthetic layout if spans is not given).
        """

        if spans is None:
            spans = []
            pos = 0
            for w in words:
                spans.append((pos, pos + len(w)))
                pos += len(w) + 1

        self._limit_hit = False
        buckets = {0: [_FragState(0.0, 0.0, (), ())]}
        n = len(words)
        for pos in range(n):
            bucket = buckets.get(pos)
            if not bucket:
                continue
            word = words[pos]
            start, end = spans[pos]
            for state in bucket:
                # Keep the word as-is (free if it is in the dictionary).
                keep_cost = 0.0 if word in self.word_set else self.oov_cost
                self._advance(buckets, pos + 1, state, keep_cost, (word,),
                              None)
                # Replace the word with a dictionary word.
                cands, trunc = self._word_candidates(word)
                self._limit_hit = self._limit_hit or trunc
                for c, w in cands:
                    change = (start, end, word, w, "spell", c)
                    self._advance(buckets, pos + 1, state, c, (w,), change)
                # Split the word into two dictionary words.
                if self.allow_split:
                    cands, trunc = self._split_candidates(word)
                    self._limit_hit = self._limit_hit or trunc
                    for c, (w1, w2) in cands:
                        change = (start, end, word, w1 + " " + w2, "split", c)
                        self._advance(buckets, pos + 1, state, c, (w1, w2),
                                      change)
                # Merge this word and the next into one dictionary word.
                if self.allow_merge and pos + 1 < n:
                    cands, trunc = self._merge_candidates(word, words[pos + 1])
                    self._limit_hit = self._limit_hit or trunc
                    end2 = spans[pos + 1][1]
                    for c, w in cands:
                        original = word + " " + words[pos + 1]
                        change = (start, end2, original, w, "merge", c)
                        self._advance(buckets, pos + 2, state, c, (w,),
                                      change)
            self._prune(buckets.get(pos + 1, []))
            if pos + 2 <= n:
                self._prune(buckets.get(pos + 2, []))

        finals = buckets.get(n, [])
        finals.sort(key=self._rank_key)
        # Selecting the top-K final states is the intended output limit,
        # not a search limit, so it does not affect exhaustiveness.
        finals = finals[: self.top_k]

        alternatives = []
        for state in finals:
            changes = tuple(
                QueryChange(s, e, orig, repl, self._reason(kind, orig, repl, c))
                for s, e, orig, repl, kind, c in state.changes
            )
            score = state.cost - self.context_weight * state.context
            alternatives.append(
                RewrittenQuery(" ".join(state.out), state.cost,
                               state.context, score, changes)
            )
        return alternatives, not self._limit_hit

    @staticmethod
    def _reason(kind, original, replacement, cost):
        if kind == "spell":
            return "corrected spelling of '%s' to '%s' (cost %g)" % (
                original, replacement, cost)
        elif kind == "split":
            return "split '%s' into '%s' (cost %g)" % (
                original, replacement, cost)
        else:
            return "merged '%s' into '%s' (cost %g)" % (
                original, replacement, cost)

    # Whole-query correction

    def correct(self, qstring, limit=None):
        """
        Corrects a raw user query string and returns a QueryCorrection
        object. Only runs of plain words are corrected; query syntax (field:
        prefixes, quoted phrases, boolean operators, wildcards, boosts and
        groups) is preserved verbatim.

        :param qstring: the raw user query string.
        :param limit: the maximum number of alternatives to return
            (defaults to top_k).
        """

        limit = limit or self.top_k
        fragments = _scan_fragments(qstring)
        per_fragment = []
        exhaustive = True
        for frag in fragments:
            words = [w for _, _, w in frag]
            spans = [(s, e) for s, e, _ in frag]
            alts, frag_exhaustive = self.correct_fragment(words, spans)
            exhaustive = exhaustive and frag_exhaustive
            per_fragment.append(alts)

        alternatives = []
        if per_fragment:
            # Combine the per-fragment alternatives: the k-th overall
            # alternative uses the k-th best alternative of each fragment
            # (falling back to the best one when a fragment has fewer).
            count = min(max(len(alts) for alts in per_fragment), limit)
            for k in range(count):
                changes = []
                cost = 0.0
                context = 0.0
                for alts in per_fragment:
                    alt = alts[min(k, len(alts) - 1)]
                    changes.extend(alt.changes)
                    cost += alt.cost
                    context += alt.context_score
                changes = tuple(
                    sorted(changes, key=lambda c: (c.start, c.end)))
                text = _apply_changes(qstring, changes)
                score = cost - self.context_weight * context
                alternatives.append(
                    RewrittenQuery(text, cost, context, score, changes))
            alternatives.sort(key=lambda a: (a.score, a.text))
        else:
            alternatives.append(RewrittenQuery(qstring, 0.0, 0.0, 0.0, ()))

        return QueryCorrection(qstring, alternatives, exhaustive)
