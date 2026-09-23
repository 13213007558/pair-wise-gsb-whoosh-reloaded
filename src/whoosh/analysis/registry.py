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
A process-wide registry of named, versioned analyzers.

Field types such as :class:`whoosh.fields.MultiLangTEXT` store the *names*
and *versions* of the analyzers they use as part of the index configuration
(the schema is pickled into the index). When the index is opened later, the
stored names/versions are checked against this registry so that a changed or
missing analyzer implementation is reported clearly instead of silently
producing different tokens.
"""

# Maps name -> (analyzer, version)
_ANALYZER_REGISTRY = {}


def register_analyzer(name, analyzer, version=1):
    """
    Registers an analyzer object under a name, with a version identifier.

    The ``version`` can be any hashable, comparable object (usually an
    integer or a string). Bump it whenever the analysis chain changes in a
    way that would produce different tokens for the same input. Indexes that
    were created with a different version of the same named analyzer will
    refuse to open, instead of silently mixing incompatible tokens.

    Registering the same name again overwrites the previous entry.

    :param name: the name to register the analyzer under.
    :param analyzer: the :class:`whoosh.analysis.Analyzer` object.
    :param version: a version identifier for this implementation of the
        analyzer.
    """

    if not name or not isinstance(name, str):
        raise ValueError(f"Analyzer name must be a non-empty string, got {name!r}")
    if not callable(analyzer):
        raise ValueError(f"{analyzer!r} is not an analyzer")
    _ANALYZER_REGISTRY[name] = (analyzer, version)


def unregister_analyzer(name):
    """
    Removes the analyzer registered under the given name, if any.
    """

    _ANALYZER_REGISTRY.pop(name, None)


def is_registered(name):
    """
    Returns True if an analyzer is registered under the given name.
    """

    return name in _ANALYZER_REGISTRY


def registered_analyzer(name):
    """
    Returns the analyzer registered under the given name.

    :raises KeyError: if no analyzer is registered under the name.
    """

    try:
        return _ANALYZER_REGISTRY[name][0]
    except KeyError:
        raise KeyError(f"No analyzer is registered under the name {name!r}")


def registered_version(name):
    """
    Returns the version identifier of the analyzer registered under the
    given name.

    :raises KeyError: if no analyzer is registered under the name.
    """

    try:
        return _ANALYZER_REGISTRY[name][1]
    except KeyError:
        raise KeyError(f"No analyzer is registered under the name {name!r}")


def clear_registry():
    """
    Removes all registered analyzers. Mostly useful for testing.
    """

    _ANALYZER_REGISTRY.clear()
