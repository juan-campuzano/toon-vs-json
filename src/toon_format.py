"""
Toon Format (Token-Optimized Object Notation) encoder and decoder.

Toon is a compact serialization format designed to reduce token usage when
passing structured data to/from LLMs, while remaining human-readable and
fully round-trippable to/from JSON.

Differences from JSON
---------------------
* Object keys are unquoted when they consist only of word characters (``\\w``).
* String values are unquoted when they contain no special characters
  (``{}``, ``[]``, ``:``, ``,``, ``'``, ``\\``).
* Booleans are encoded as ``T`` / ``F`` instead of ``true`` / ``false``.
* ``null`` is encoded as ``~``.
* Whitespace between structural tokens is omitted.

Example
-------
JSON  : {"name": "Alice", "age": 30, "active": true, "tags": ["a", "b"]}
Toon  : {name:Alice,age:30,active:T,tags:[a,b]}
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["dumps", "loads", "to_toon", "from_toon"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Characters that force a string value to be single-quoted
_SPECIAL_CHARS_RE = re.compile(r'[{}[\]:,\'\\]')
# A key is safe to emit without quotes when it only contains word chars
_SAFE_KEY_RE = re.compile(r'^\w+$')
# A value is safe to emit bare only when it looks like a plain identifier
# (starts with a letter or underscore, followed by word chars).
# This avoids the lexer mis-tokenising strings that start with digits
# (e.g. "2024-01-15") or contain dots/hyphens adjacent to digits.
_SAFE_VALUE_RE = re.compile(r'^[A-Za-z_]\w*$')
# Tokens produced by the lexer
_TOKEN_RE = re.compile(
    r"""
      (?P<lbrace>   \{               )
    | (?P<rbrace>   \}               )
    | (?P<lbracket> \[               )
    | (?P<rbracket> \]               )
    | (?P<colon>    :                )
    | (?P<comma>    ,                )
    | (?P<true>     T(?=[ ,}\]:]|$)  )
    | (?P<false>    F(?=[ ,}\]:]|$)  )
    | (?P<null>     ~(?=[ ,}\]:]|$)  )
    | (?P<number>   -?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?  )
    | (?P<quoted>   '(?:[^'\\]|\\.)*')
    | (?P<bare>     [^{}\[\]:,'\s]+  )
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _encode_value(obj: Any) -> str:
    """Recursively encode a Python object into Toon notation."""
    if obj is None:
        return "~"
    if isinstance(obj, bool):
        return "T" if obj else "F"
    if isinstance(obj, (int, float)):
        # Use JSON's number representation for safety
        return json.dumps(obj)
    if isinstance(obj, str):
        return _encode_str(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode_value(v) for v in obj) + "]"
    if isinstance(obj, dict):
        pairs = []
        for k, v in obj.items():
            encoded_key = k if _SAFE_KEY_RE.match(k) else "'" + k.replace("\\", "\\\\").replace("'", "\\'") + "'"
            pairs.append(f"{encoded_key}:{_encode_value(v)}")
        return "{" + ",".join(pairs) + "}"
    # Fallback for other types (e.g., datetime)
    return _encode_str(str(obj))


def _encode_str(s: str) -> str:
    """Encode a string, quoting it only when necessary."""
    if s == "":
        return "''"
    # Use _SAFE_VALUE_RE as the single gate: only plain identifiers (starting
    # with a letter or underscore, containing only word chars) are emitted bare.
    # Everything else is single-quoted.  This prevents the lexer from
    # mis-tokenising strings that contain digits, hyphens, dots, spaces, etc.
    if _SAFE_VALUE_RE.match(s) and s not in ("T", "F"):
        return s
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return "'" + escaped + "'"


def dumps(obj: Any) -> str:
    """Serialize *obj* to a Toon-formatted string."""
    return _encode_value(obj)


# Alias
to_toon = dumps


# ---------------------------------------------------------------------------
# Decoder / parser
# ---------------------------------------------------------------------------


class _ParseError(ValueError):
    pass


class _Parser:
    """Recursive-descent parser for Toon notation."""

    def __init__(self, text: str) -> None:
        self._tokens = list(_TOKEN_RE.finditer(text.strip()))
        self._pos = 0

    # -- token helpers ------------------------------------------------------

    def _peek(self) -> re.Match | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, kind: str | None = None) -> re.Match:
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end of input")
        if kind is not None and tok.lastgroup != kind:
            raise _ParseError(
                f"Expected {kind!r} but got {tok.lastgroup!r} ({tok.group()!r})"
            )
        self._pos += 1
        return tok

    # -- value parsers ------------------------------------------------------

    def parse_value(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end of input")
        kind = tok.lastgroup
        if kind == "lbrace":
            return self.parse_object()
        if kind == "lbracket":
            return self.parse_array()
        if kind == "true":
            self._consume("true")
            return True
        if kind == "false":
            self._consume("false")
            return False
        if kind == "null":
            self._consume("null")
            return None
        if kind == "number":
            self._consume("number")
            return json.loads(tok.group())
        if kind == "quoted":
            self._consume("quoted")
            raw = tok.group()[1:-1]  # strip surrounding quotes
            return raw.replace("\\'", "'").replace("\\\\", "\\")
        if kind == "bare":
            self._consume("bare")
            return tok.group()
        raise _ParseError(f"Unexpected token {tok.group()!r}")

    def parse_object(self) -> dict:
        self._consume("lbrace")
        result: dict = {}
        if self._peek() and self._peek().lastgroup == "rbrace":
            self._consume("rbrace")
            return result
        while True:
            # key
            tok = self._peek()
            if tok is None:
                raise _ParseError("Unexpected end of input in object")
            if tok.lastgroup == "quoted":
                self._consume("quoted")
                key = tok.group()[1:-1].replace("\\'", "'").replace("\\\\", "\\")
            elif tok.lastgroup == "bare":
                self._consume("bare")
                key = tok.group()
            else:
                raise _ParseError(f"Expected object key, got {tok.group()!r}")
            self._consume("colon")
            value = self.parse_value()
            result[key] = value
            nxt = self._peek()
            if nxt is None or nxt.lastgroup == "rbrace":
                break
            self._consume("comma")
        self._consume("rbrace")
        return result

    def parse_array(self) -> list:
        self._consume("lbracket")
        result: list = []
        if self._peek() and self._peek().lastgroup == "rbracket":
            self._consume("rbracket")
            return result
        while True:
            result.append(self.parse_value())
            nxt = self._peek()
            if nxt is None or nxt.lastgroup == "rbracket":
                break
            self._consume("comma")
        self._consume("rbracket")
        return result

    def parse(self) -> Any:
        if not self._tokens:
            raise _ParseError("Empty input")
        value = self.parse_value()
        if self._peek() is not None:
            raise _ParseError(
                f"Trailing data after value: {self._peek().group()!r}"
            )
        return value


def loads(text: str) -> Any:
    """Deserialize a Toon-formatted string to a Python object."""
    return _Parser(text).parse()


# Alias
from_toon = loads
