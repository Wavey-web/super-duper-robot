"""
Python 3.10+ compatibility shim.

collections.MutableMapping (and friends) were removed from the
top-level `collections` module in Python 3.10. They were moved to
`collections.abc` in Python 3.3. Many third-party packages still
reference `collections.MutableMapping`, so we monkey-patch them back.

Import this module early (before any third-party imports) in every
entry point to ensure compatibility.
"""

import collections
import collections.abc

for _attr in (
    'MutableMapping', 'MutableSequence', 'MutableSet',
    'Mapping', 'Sequence', 'Set', 'Callable', 'Iterable',
    'Iterator', 'MutableSet',
):
    if not hasattr(collections, _attr) and hasattr(collections.abc, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))
