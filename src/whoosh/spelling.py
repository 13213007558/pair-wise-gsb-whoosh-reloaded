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

import re
from bisect import bisect_left
from heapq import heappush, heapreplace
from math import log1p

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


# Segment-level correction (split/merge + context)


class EditCosts:
    """
    Configurable costs for the edit operations used by
    :class:`SegmentCorrector`.

    :param insert: cost of inserting a letter.
    :param delete: cost of deleting a letter.
    :param replace: cost of replacing one letter with another.
    :param split: cost of splitting one word into two words (added to any
        letter-level costs of correcting the two parts).
    :param merge: cost of merging two adjacent words into one word (added to
        any letter-level cost of correcting the joined word).
    """

    def __init__(self, insert=1.0, delete=1.0, replace=1.0, split=1.0, merge=1.0):
        self.insert = insert
        self.delete = delete
        self.replace = replace
        self.split = split
        self.merge = merge

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(insert={self.insert!r}, "
            f"delete={self.delete!r}, replace={self.replace!r}, "
            f"split={self.split!r}, merge={self.merge!r})"
        )


def weighted_distance(a, b, costs=None, max_cost=None):
    """
    Returns the weighted edit distance between strings ``a`` and ``b``
    using the insert/delete/replace costs from an :class:`EditCosts` object.
    If ``max_cost`` is given, the function may return early with a value
    greater than ``max_cost`` as soon as it is known the distance exceeds
    it.
    """

    if costs is None:
        costs = EditCosts()
    ins, dele, rep = costs.insert, costs.delete, costs.replace
    if max_cost is None:
        max_cost = float("inf")

    lb = len(b)
    prev = [j * ins for j in range(lb + 1)]
    for i in range(1, len(a) + 1):
        ca = a[i - 1]
        cur = [i * dele] + [0.0] * lb
        rowmin = cur[0]
        for j in range(1, lb + 1):
            sub = prev[j - 1] + (0.0 if ca == b[j - 1] else rep)
            best = min(sub, prev[j] + dele, cur[j - 1] + ins)
            cur[j] = best
            if best < rowmin:
                rowmin = best
        if rowmin > max_cost:
            # Row minimums never decrease, so the final distance is
            # definitely over the budget
            return rowmin
        prev = cur
    return prev[lb]


class Lexicon:
    """
    A read-only dictionary of known words with optional frequencies, used as
    the source of correction candidates. Can be built from an index reader
    (the index is only read, never written) or from a plain word list.
    """

    def __init__(self, words, frequencies=None):
        """
        :param words: an iterable of word strings.
        :param frequencies: an optional mapping of word -> frequency.
        """

        self._words = sorted(set(words))
        self._wordset = frozenset(self._words)
        self._freqs = dict(frequencies) if frequencies else {}

    @classmethod
    def from_reader(cls, reader, fieldname):
        """
        Builds a lexicon from the terms of a field in an index reader, using
        the field frequencies as word frequencies. This only reads the
        dictionary of the index.
        """

        words = []
        freqs = {}
        for term in reader.lexicon(fieldname):
            word = term.decode("utf8") if isinstance(term, bytes) else term
            words.append(word)
            freqs[word] = reader.frequency(fieldname, word)
        return cls(words, freqs)

    def __contains__(self, word):
        return word in self._wordset

    def __len__(self):
        return len(self._words)

    def words(self):
        """Returns the sorted list of words in the lexicon."""

        return self._words

    def frequency(self, word):
        """Returns the frequency of a word, or 0 if it is not in the lexicon."""

        return self._freqs.get(word, 0)


class Edit:
    """
    Represents a single change made to the original string. Has the
    following attributes:

    ``start``/``end``
        The character span of the changed text in the **original** string.
    ``original``
        The original text of the span.
    ``replacement``
        The text the span was replaced with.
    ``reason``
        Why the change was made: one of ``"spell"``, ``"split"``,
        ``"merge"``, ``"spell+split"``, or ``"spell+merge"``.
    """

    __slots__ = ("start", "end", "original", "replacement", "reason")

    def __init__(self, start, end, original, replacement, reason):
        self.start = start
        self.end = end
        self.original = original
        self.replacement = replacement
        self.reason = reason

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.start!r}, {self.end!r}, "
            f"{self.original!r}, {self.replacement!r}, {self.reason!r})"
        )

    def __eq__(self, other):
        return (
            isinstance(other, Edit)
            and self.start == other.start
            and self.end == other.end
            and self.original == other.original
            and self.replacement == other.replacement
            and self.reason == other.reason
        )


class Rewrite:
    """
    One suggested rewrite of a query string or plain-text segment. Has the
    following attributes:

    ``text``
        The rewritten string.
    ``cost``
        The total edit cost of the rewrite.
    ``context``
        The context score of the rewrite (higher means the word sequence was
        more likely according to the supplied adjacent-word frequencies).
    ``edits``
        A list of :class:`Edit` objects locating each change in the
        original string.
    """

    __slots__ = ("text", "cost", "context", "edits")

    def __init__(self, text, cost, context, edits):
        self.text = text
        self.cost = cost
        self.context = context
        self.edits = edits

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.text!r}, cost={self.cost!r}, "
            f"context={self.context!r})"
        )

    @property
    def score(self):
        # Lower scores are better; rewrites are sorted by this value
        return self.cost - self.context


class SegmentCorrection:
    """
    The result of correcting a query string or plain-text segment with
    :class:`SegmentCorrector`. Has the following attributes:

    ``original``
        The original string.
    ``rewrites``
        A list of up to K :class:`Rewrite` objects, best first.
    ``exhaustive``
        False if the candidate or search-state limits were hit, meaning the
        search was not exhaustive and better rewrites may exist.
    """

    def __init__(self, original, rewrites, exhaustive=True):
        self.original = original
        self.rewrites = rewrites
        self.exhaustive = exhaustive

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.original!r}, "
            f"{self.rewrites!r}, exhaustive={self.exhaustive!r})"
        )

    @property
    def text(self):
        """The text of the best rewrite, or the original string."""

        return self.rewrites[0].text if self.rewrites else self.original

    @property
    def edits(self):
        """The edits of the best rewrite."""

        return self.rewrites[0].edits if self.rewrites else []


# Query string tokenizing

# A field name prefix such as "title:" or "sku:"
_field_prefix_re = re.compile(r"^([A-Za-z_][\w.]*):(.*)$", re.S)
# Tokens that are always treated as query syntax
_operator_words = frozenset(["AND", "OR", "NOT", "&&", "||", "!"])
# Characters that mark a token as query syntax (wildcards, fuzzy, boosts)
_syntax_chars = frozenset("*?~^")


def tokenize_query_string(qstring):
    """
    Splits a query string into a list of ``(kind, text, start, end)`` token
    tuples. ``kind`` is either ``"term"`` (a plain word that may be
    corrected) or ``"syntax"`` (quoted phrases, field prefixes, boolean
    operators, wildcards, and other query syntax that must be preserved
    verbatim).
    """

    tokens = []
    i, n = 0, len(qstring)
    while i < n:
        ch = qstring[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            # Quoted phrase: protected up to the closing quote
            j = qstring.find('"', i + 1)
            j = n if j < 0 else j + 1
            tokens.append(("syntax", qstring[i:j], i, j))
            i = j
            continue
        if ch in "()[]{}":
            tokens.append(("syntax", ch, i, i + 1))
            i += 1
            continue
        # Read a whitespace-delimited token
        j = i
        while j < n and not qstring[j].isspace() and qstring[j] not in '()[]{}"':
            j += 1
        tok = qstring[i:j]
        if (
            tok in _operator_words
            or tok[0] in "+-!"
            or any(c in _syntax_chars for c in tok)
        ):
            tokens.append(("syntax", tok, i, j))
        else:
            m = _field_prefix_re.match(tok)
            if m:
                # The "field:" prefix is syntax; the word after it may be
                # corrected unless it itself contains syntax characters
                prefix, rest = m.group(1), m.group(2)
                cut = i + len(prefix) + 1
                tokens.append(("syntax", qstring[i:cut], i, cut))
                if rest:
                    if any(c in _syntax_chars for c in rest):
                        tokens.append(("syntax", rest, cut, j))
                    else:
                        tokens.append(("term", rest, cut, j))
            else:
                tokens.append(("term", tok, i, j))
        i = j
    return tokens


def _apply_edits(original, edits):
    # Rebuilds a string from the original text and a sequence of
    # (start, end, original, replacement, reason) edit tuples
    out = []
    pos = 0
    for edit in sorted(edits, key=lambda e: e[0]):
        out.append(original[pos : edit[0]])
        out.append(edit[3])
        pos = edit[1]
    out.append(original[pos:])
    return "".join(out)


class SegmentCorrector:
    """
    Corrects whole query fragments instead of single words. In addition to
    letter-level insert/delete/replace edits, this corrector can **split**
    one word into two dictionary words and **merge** two adjacent words into
    one dictionary word, and it ranks whole-fragment rewrites using
    application-supplied adjacent-word (bigram) frequencies as context.

    Candidates come only from a :class:`Lexicon` (which may be built from
    an index reader) and the supplied bigram frequencies; no online models
    are used.

    :param lexicon: a :class:`Lexicon` (or any object with ``words()``,
        ``frequency(word)``, and ``__contains__``).
    :param bigram_freq: adjacent-word frequencies used for context scoring,
        either a mapping of ``(word1, word2)`` -> count or a callable
        taking two words and returning a count.
    :param costs: an :class:`EditCosts` object with the configurable
        insert/delete/replace/split/merge costs.
    :param context_weight: how strongly the context score influences the
        ranking. The ranking score is ``cost - context``, where
        ``context`` is this weight times the sum of
        ``log1p(bigram_freq(prev, word))`` over the rewritten word
        sequence.
    :param max_word_cost: the maximum total letter-level edit cost allowed
        for correcting a single word (or the parts of a split, or a merged
        word).
    :param max_candidates: the maximum number of replacement, split, or merge
        candidates kept per word. If more candidates exist, the result is
        marked as not exhaustive.
    :param max_states: the maximum number of search states kept per input
        position. If more states exist, the best are kept and the result is
        marked as not exhaustive.
    :param allow_split: allow splitting one word into two words.
    :param allow_merge: allow merging two adjacent words into one word. When
        both ``allow_split`` and ``allow_merge`` are False, the
        corrector degrades to classic per-word spelling suggestions.
    :param unknown_cost: the cost of keeping a word that does not appear in
        the lexicon. Keeping a dictionary word is always free. The default
        makes any affordable correction beat keeping an unknown word; set it
        to 0 to rank purely by edit cost and context.
    """

    def __init__(
        self,
        lexicon,
        bigram_freq=None,
        costs=None,
        context_weight=1.0,
        max_word_cost=2.0,
        max_candidates=8,
        max_states=256,
        allow_split=True,
        allow_merge=True,
        unknown_cost=3.0,
    ):
        self.lexicon = lexicon
        self.costs = costs or EditCosts()
        self.context_weight = context_weight
        self.max_word_cost = max_word_cost
        self.max_candidates = max_candidates
        self.max_states = max_states
        self.allow_split = allow_split
        self.allow_merge = allow_merge
        self.unknown_cost = unknown_cost

        if bigram_freq is None:
            self._bigram = lambda w1, w2: 0
        elif callable(bigram_freq):
            self._bigram = bigram_freq
        else:
            self._bigram = lambda w1, w2: bigram_freq.get((w1, w2), 0)

    # Candidate generation

    def _corrections(self, word, max_cost):
        # Returns (list of (cost, word), truncated) for dictionary words
        # within max_cost of the given word, excluding the word itself
        lex = self.lexicon
        cands = []
        for w in lex.words():
            if w == word:
                continue
            d = weighted_distance(word, w, self.costs, max_cost)
            if d <= max_cost:
                cands.append((d, w))
        cands.sort(key=lambda cw: (cw[0], -lex.frequency(cw[1]), cw[1]))
        truncated = len(cands) > self.max_candidates
        return cands[: self.max_candidates], truncated

    def _corrections_incl(self, word):
        # Like _corrections but includes the word itself at cost 0 if it is
        # a dictionary word
        cands, truncated = self._corrections(word, self.max_word_cost)
        if word in self.lexicon:
            cands = [(0.0, word)] + cands
        return cands, truncated

    def _split_options(self, word):
        # Returns (list of (cost, reason, (w1, w2), replacement), truncated)
        truncated = False
        best = {}
        for i in range(1, len(word)):
            left, right = word[:i], word[i:]
            lcands, trunc = self._corrections_incl(left)
            truncated = truncated or trunc
            rcands, trunc = self._corrections_incl(right)
            truncated = truncated or trunc
            for lcost, lw in lcands:
                for rcost, rw in rcands:
                    cost = lcost + rcost
                    if cost > self.max_word_cost:
                        continue
                    key = (lw, rw)
                    if key not in best or cost < best[key]:
                        best[key] = cost

        freq = self.lexicon.frequency
        split = self.costs.split
        items = sorted(
            best.items(),
            key=lambda kv: (
                split + kv[1],
                -(freq(kv[0][0]) + freq(kv[0][1])),
                kv[0],
            ),
        )
        if len(items) > self.max_candidates:
            truncated = True
            items = items[: self.max_candidates]

        options = []
        for (lw, rw), cost in items:
            reason = "split" if cost == 0 else "spell+split"
            options.append((split + cost, reason, (lw, rw), lw + " " + rw))
        return options, truncated

    def _merge_options(self, word1, word2):
        # Returns (list of (cost, reason, (word,), replacement), truncated)
        joined = word1 + word2
        cands, truncated = self._corrections_incl(joined)
        options = []
        for cost, w in cands:
            reason = "merge" if w == joined else "spell+merge"
            options.append((self.costs.merge + cost, reason, (w,), w))
        return options, truncated

    def _single_options(self, word):
        # Returns (list of (cost, reason, out_words, replacement), truncated)
        # for keeping, spelling-correcting, or splitting a single word.
        # A replacement of None means the word is kept unchanged.
        keep_cost = 0.0 if word in self.lexicon else self.unknown_cost
        options = [(keep_cost, "keep", (word,), None)]
        cands, truncated = self._corrections(word, self.max_word_cost)
        seen = {word}
        for cost, w in cands:
            if w not in seen:
                seen.add(w)
                options.append((cost, "spell", (w,), w))
        if self.allow_split and len(word) > 1:
            splits, trunc = self._split_options(word)
            truncated = truncated or trunc
            for cost, reason, words, replacement in splits:
                if words not in seen:
                    seen.add(words)
                    options.append((cost, reason, words, replacement))
        return options, truncated

    # Search

    def _context_add(self, last_word, out_words):
        add = 0.0
        prev = last_word
        for w in out_words:
            if prev is not None:
                add += log1p(self._bigram(prev, w))
            prev = w
        return add * self.context_weight

    def _dedupe_beam(self, states):
        # Dedupe states producing identical output, keep the best-scoring
        # variant, then keep at most max_states states
        bykey = {}
        for state in states:
            cost, context, last, edits = state
            key = (last, edits)
            if key not in bykey or cost - context < bykey[key][0] - bykey[key][1]:
                bykey[key] = state
        states = sorted(
            bykey.values(),
            key=lambda s: (s[0] - s[1], s[2] or "", s[3]),
        )
        truncated = len(states) > self.max_states
        return states[: self.max_states], truncated

    def _run(self, tokens, original, k):
        # A state is a (cost, context, last_word, edits) tuple, where edits
        # is a tuple of (start, end, original, replacement, reason) tuples
        n = len(tokens)
        positions = {0: [(0.0, 0.0, None, ())]}
        exhaustive = True
        final = []

        for i in range(n + 1):
            states = positions.get(i)
            if not states:
                continue
            if i == n:
                final = states
                break

            kind, text, start, end = tokens[i]
            if kind == "syntax":
                # Syntax tokens are preserved verbatim and reset the context
                nxt = positions.setdefault(i + 1, [])
                for cost, context, _, edits in states:
                    nxt.append((cost, context, None, edits))
            else:
                options, trunc = self._single_options(text)
                exhaustive = exhaustive and not trunc

                merge_opts = []
                if self.allow_merge and i + 1 < n and tokens[i + 1][0] == "term":
                    merge_opts, trunc = self._merge_options(text, tokens[i + 1][1])
                    exhaustive = exhaustive and not trunc

                nxt = positions.setdefault(i + 1, [])
                nxt2 = positions.setdefault(i + 2, []) if merge_opts else None
                for cost, context, last, edits in states:
                    for ocost, reason, out_words, replacement in options:
                        ctx_add = self._context_add(last, out_words)
                        new_edits = edits
                        if replacement is not None:
                            new_edits = edits + (
                                (
                                    start,
                                    end,
                                    original[start:end],
                                    replacement,
                                    reason,
                                ),
                            )
                        nxt.append(
                            (
                                cost + ocost,
                                context + ctx_add,
                                out_words[-1],
                                new_edits,
                            )
                        )
                    for ocost, reason, out_words, replacement in merge_opts:
                        ctx_add = self._context_add(last, out_words)
                        mend = tokens[i + 1][3]
                        new_edits = edits + (
                            (
                                start,
                                mend,
                                original[start:mend],
                                replacement,
                                reason,
                            ),
                        )
                        nxt2.append(
                            (
                                cost + ocost,
                                context + ctx_add,
                                out_words[-1],
                                new_edits,
                            )
                        )

            # Position i+1 is complete once position i has been expanded
            if i + 1 in positions:
                positions[i + 1], trunc = self._dedupe_beam(positions[i + 1])
                exhaustive = exhaustive and not trunc

        # Rank the terminal states; break ties deterministically by text
        ranked = []
        for cost, context, _, edits in final:
            text = _apply_edits(original, edits)
            ranked.append((cost - context, text, cost, context, edits))
        ranked.sort(key=lambda item: (item[0], item[1]))

        rewrites = [
            Rewrite(
                text,
                cost,
                context,
                [
                    Edit(start, end, orig, repl, reason)
                    for start, end, orig, repl, reason in edits
                ],
            )
            for _, text, cost, context, edits in ranked[:k]
        ]
        return SegmentCorrection(original, rewrites, exhaustive)

    # Public API

    def word_suggestions(self, word, limit=5, max_cost=None):
        """
        Returns per-word spelling suggestions for a single word, mirroring
        the behavior of :meth:`Corrector.suggest`: a list of dictionary
        words within ``max_cost`` of the given word, best first. The word
        itself is never included.
        """

        if max_cost is None:
            max_cost = self.max_word_cost
        cands, _ = self._corrections(word, max_cost)
        return [w for _, w in cands[:limit]]

    def suggest_segment(self, text, k=3):
        """
        Returns a :class:`SegmentCorrection` with the top K rewrites of a
        plain-text fragment (no query syntax).
        """

        tokens = [
            ("term", m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)
        ]
        return self._run(tokens, text, k)

    def correct_string(self, qstring, k=3):
        """
        Returns a :class:`SegmentCorrection` with the top K rewrites of a
        query string. Quoted phrases, field prefixes, boolean operators,
        wildcards, and other query syntax are preserved verbatim; only plain
        word fragments are corrected.
        """

        return self._run(tokenize_query_string(qstring), qstring, k)
