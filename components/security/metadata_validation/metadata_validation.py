"""Validate document metadata against an explicit, declared schema.

Why metadata is a security surface
----------------------------------
Metadata looks like harmless bookkeeping -- ``{"page": 3, "lang": "en"}`` --
and gets treated that way. Then it gets rendered in a UI, written to a log,
search-filtered on, and, very often, concatenated into a prompt so the model
can "use the context". At that point a metadata value is a string that reaches
a model, and nobody validated it on the way in.

That is how ``{"source": "docs.md\\n\\nIgnore all previous instructions"}``
becomes an attack. The retriever did its job. The chunk was fine. The payload
rode in on a field nobody was checking.

So: **metadata is untrusted data**, and this component treats it that way.

Reject, because you own the keys
--------------------------------
This component **rejects** rather than flags, which is the opposite of what
:mod:`retrieved_content_safety` does. The difference is deliberate and it is
the whole design:

- **Metadata is a controlled vocabulary.** The application writes these keys.
  No legitimate value contains "ignore all previous instructions", so a match
  is either a bug or an attack -- and neither should be passed through with a
  note attached.
- **Retrieved content is genuinely untrusted.** It legitimately quotes things,
  including instructions, because that is what documents *are*. Flagging is
  right there; deleting would be lossy and still bypassable.

Rejecting here also means the failure is loud and early, at the point where a
developer can see which field caused it -- rather than as a subtly wrong
answer three stages later.

The schema is declared, not inferred
------------------------------------
A validator that accepts "anything shaped vaguely like a dict" validates
nothing. Every field is declared: its type, an optional length bound, an
optional set of allowed values, whether it is required. Undeclared keys are
rejected unless ``allow_extra=True`` is set **explicitly** -- the default is
the safe one, so a new field cannot appear by accident and travel unvalidated.

Types are checked strictly. ``True`` is not accepted where an ``int`` is
declared, because in Python ``bool`` is a subclass of ``int`` and that one
concession is how a type check gets quietly defeated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Final

__all__ = [
    "DEFAULT_INJECTION_PATTERNS",
    "DEFAULT_MAX_KEYS",
    "Field",
    "INJECTION_PATTERNS",
    "MetadataError",
    "MetadataSchema",
    "MetadataValueError",
    "SchemaViolation",
]

DEFAULT_MAX_KEYS: Final[int] = 32

#: Patterns that mean "this string is talking to the reader's *system*, not
#: describing a document". Each is a real, published prompt-injection shape.
#: Lowercased and matched as substrings, because casing is not a defence.
INJECTION_PATTERNS: Final[tuple[str, ...]] = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard all",
    "forget your instructions",
    "forget everything",
    "you are now",
    "you're now",
    "new instructions:",
    "system prompt",
    "developer message",
    "act as if",
    "role-play as",
    "do not tell the user",
    "don't tell the user",
    "without telling the user",
    "reveal your instructions",
    "output your instructions",
)

#: Compiled once. ``(?s)`` is not used: these are substring matches, and a
#: ``.*`` across lines would make a single match swallow a whole document.
DEFAULT_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(re.escape(phrase)) for phrase in INJECTION_PATTERNS
)

_MISSING: Any = object()

#: Keys are interpolated into filters, filenames and prompts, so they get a
#: narrow character set rather than a screen. Identical posture to
#: ``InputValidator.validate_identifier``, held locally because this component
#: copies out on its own.
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class MetadataError(Exception):
    """Base for every failure in this component."""


class SchemaViolation(MetadataError):
    """The metadata did not match its declared schema."""


class MetadataValueError(SchemaViolation):
    """A *value* was well-formed but hostile -- an injection payload."""


@dataclass(frozen=True)
class Field:
    """One declared metadata key.

    ``type`` is checked strictly: a ``bool`` is not an ``int`` here, even
    though Python says it is. ``choices`` restricts the value to a set, which
    is the strongest thing you can do -- an allow-list needs no screening
    because nothing outside it gets through.
    """

    type: type | tuple[type, ...] = str
    max_length: int | None = None
    choices: frozenset[Any] | None = None
    required: bool = False
    default: Any = dataclass_field(default=_MISSING)

    def has_default(self) -> bool:
        return self.default is not _MISSING


def _describe(value: object) -> str:
    return f"{type(value).__name__} ({value!r})"


class MetadataSchema:
    """Validate a metadata mapping against declared fields.

    ``allow_extra`` defaults to ``False`` on purpose. An undeclared key is one
    nobody has type-checked, length-bounded or screened, so the safe default
    is to refuse it -- a new field should have to be declared before it can
    travel, not discovered in a log later.

    ``max_keys`` bounds the mapping itself. Pathological metadata is a
    different attack from hostile metadata, and it needs its own ceiling.
    """

    def __init__(
        self,
        *,
        fields: Mapping[str, Field],
        allow_extra: bool = False,
        max_keys: int = DEFAULT_MAX_KEYS,
        patterns: Sequence[re.Pattern[str]] = DEFAULT_INJECTION_PATTERNS,
        name: str = "metadata-validation",
    ) -> None:
        if not isinstance(fields, Mapping) or not fields:
            raise MetadataError(
                f"metadata-validation: fields must be a non-empty mapping of "
                f"key -> Field, got {type(fields).__name__}"
            )
        for key, spec in fields.items():
            if not isinstance(key, str) or not _KEY.match(key):
                raise MetadataError(
                    f"metadata-validation: {key!r} is not a usable field name. "
                    f"Names must match [A-Za-z][A-Za-z0-9_.-]{{0,63}} -- they are "
                    f"interpolated into filters and prompts"
                )
            if not isinstance(spec, Field):
                raise MetadataError(
                    f"metadata-validation: fields[{key!r}] is "
                    f"{type(spec).__name__}, expected a Field"
                )
            declared = (
                (spec.type,) if isinstance(spec.type, type)
                else spec.type if isinstance(spec.type, tuple)
                else None
            )
            if declared is None or not all(isinstance(t, type) for t in declared):
                raise MetadataError(
                    f"metadata-validation: fields[{key!r}].type must be a type "
                    f"or a tuple of types, got {spec.type!r}"
                )
            if spec.max_length is not None and spec.max_length < 1:
                raise MetadataError(
                    f"metadata-validation: fields[{key!r}].max_length must be "
                    f"at least 1, got {spec.max_length}"
                )
        if not isinstance(patterns, Sequence) or isinstance(patterns, (str, bytes)):
            raise MetadataError(
                f"metadata-validation: patterns must be a sequence of compiled "
                f"patterns, got {type(patterns).__name__}"
            )
        for position, pattern in enumerate(patterns):
            if not hasattr(pattern, "search") or not hasattr(pattern, "pattern"):
                raise MetadataError(
                    f"metadata-validation: patterns[{position}] is "
                    f"{type(pattern).__name__}, expected a compiled pattern"
                )
        if not isinstance(allow_extra, bool):
            raise MetadataError(
                f"metadata-validation: allow_extra must be a bool, got "
                f"{type(allow_extra).__name__}"
            )
        if isinstance(max_keys, bool) or not isinstance(max_keys, int) or max_keys < 1:
            raise MetadataError(
                f"metadata-validation: max_keys must be an int of at least 1, "
                f"got {max_keys!r}"
            )
        if not isinstance(name, str) or not name:
            raise MetadataError(
                f"metadata-validation: name must be a non-empty string, got {name!r}"
            )
        self.fields = dict(fields)
        self.allow_extra = allow_extra
        self.max_keys = max_keys
        self.patterns = tuple(patterns)
        self.name = name

    # ------------------------------------------------------------------ #
    # Schema construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_spec(cls, spec: Mapping[str, dict[str, Any]], **kwargs: Any) -> MetadataSchema:
        """Build from a plain-dict spec, e.g. loaded from JSON.

        Keeps schema definitions in data files readable by people who are not
        writing Python.
        """
        return cls(
            fields={key: Field(**value) for key, value in spec.items()},
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def errors(self, metadata: object) -> list[str]:
        """Every problem found, as readable strings. Never raises on bad data.

        The point is a report: a caller validating many records wants all the
        problems, not the first one.
        """
        if not isinstance(metadata, Mapping):
            return [
                f"metadata must be a mapping, got {type(metadata).__name__}"
            ]

        problems: list[str] = []

        if len(metadata) > self.max_keys:
            problems.append(
                f"has {len(metadata)} keys, over the {self.max_keys}-key limit"
            )

        for key in metadata:
            if not isinstance(key, str) or not _KEY.match(key):
                problems.append(f"key {key!r} is not a usable field name")

        for key, spec in self.fields.items():
            if key not in metadata:
                if spec.required and not spec.has_default():
                    problems.append(f"required field {key!r} is missing")
                continue
            problems.extend(self._check_value(key, metadata[key], spec))

        if not self.allow_extra:
            for key in metadata:
                if key not in self.fields:
                    problems.append(
                        f"undeclared field {key!r} is not permitted "
                        f"(allow_extra=False)"
                    )
        else:
            for key in metadata:
                if key not in self.fields:
                    problems.extend(
                        self._check_value(
                            key, metadata[key], Field(type=(str, int, float, bool))
                        )
                    )

        return problems

    def _check_value(self, key: str, value: Any, spec: Field) -> list[str]:
        problems: list[str] = []

        if not self._type_matches(value, spec.type):
            expected = (
                spec.type.__name__
                if isinstance(spec.type, type)
                else " or ".join(t.__name__ for t in spec.type)
            )
            problems.append(
                f"field {key!r} must be {expected}, got {_describe(value)}"
            )
            return problems

        if spec.choices is not None and value not in spec.choices:
            problems.append(
                f"field {key!r} must be one of {sorted(map(repr, spec.choices))}, "
                f"got {value!r}"
            )

        if isinstance(value, str):
            if spec.max_length is not None and len(value) > spec.max_length:
                problems.append(
                    f"field {key!r} is {len(value)} characters, over its "
                    f"{spec.max_length}-character limit"
                )
            problems.extend(self._check_injection(key, value))

        return problems

    @staticmethod
    def _type_matches(value: Any, expected: type | tuple[type, ...]) -> bool:
        types = (expected,) if isinstance(expected, type) else tuple(expected)
        # bool subclasses int; accepting that is how a type check gets
        # defeated, so bool is only accepted where bool was declared.
        if isinstance(value, bool):
            return bool in types
        return any(isinstance(value, t) for t in types)

    def _check_injection(self, key: str, value: str) -> list[str]:
        problems: list[str] = []
        for pattern in self.patterns:
            match = pattern.search(value.lower())
            if match:
                problems.append(
                    f"field {key!r} contains an instruction aimed at a model "
                    f"rather than a description of a document: matched "
                    f"{match.group(0)!r}. Metadata is concatenated into "
                    f"prompts and rendered in UIs, so this is a "
                    f"prompt-injection vector. If a legitimate value genuinely "
                    f"needs this text, widen the schema for that field "
                    f"specifically rather than loosening the check"
                )
        return problems

    def validate(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Return a new mapping with declared defaults applied.

        Returns a **copy**, never the caller's dict: a validator that returns
        the object it was handed has already given up control of it, and
        ``frozen`` metadata is a claim the caller then cannot keep.

        Raises :class:`MetadataValueError` when an injection payload was found
        and :class:`SchemaViolation` otherwise. The split is the point: one is
        a schema you got wrong, the other is a string that was aimed at your
        model, and they deserve different handling -- a bug report versus an
        alert. Both subclass ``SchemaViolation``, so a caller that does not
        care can still catch one thing.
        """
        problems = self.errors(metadata)
        if problems:
            cls = MetadataValueError if self._has_injection(metadata) else SchemaViolation
            raise cls(
                f"{self.name}: metadata failed validation with "
                f"{len(problems)} problem(s): " + "; ".join(problems)
            )

        result: dict[str, Any] = dict(metadata)
        for key, spec in self.fields.items():
            if key not in result and spec.has_default():
                result[key] = spec.default
        return result

    def _has_injection(self, metadata: object) -> bool:
        """Whether a value that was actually *checked* carried a payload.

        Mirrors :meth:`errors` exactly: declared fields, plus undeclared ones
        only when ``allow_extra`` admits them. A payload in a key that was
        rejected outright never reached the injection check, so it must not
        flip the error type to ``MetadataValueError`` -- the caller would get
        an alert saying "someone attacked you" for what is a schema error.
        """
        if not isinstance(metadata, Mapping):
            return False
        examined = [
            value
            for key, value in metadata.items()
            if isinstance(value, str)
            and (key in self.fields or self.allow_extra)
        ]
        return any(self.injection_hits(value) for value in examined)

    def is_valid(self, metadata: object) -> bool:
        """Whether ``metadata`` passes. Never raises."""
        return not self.errors(metadata)

    def injection_hits(self, value: str) -> list[str]:
        """The text ``value`` matches, without raising.

        Returns what was *matched*, not the rule that matched it -- the rule
        may be a compiled pattern, and its source form is noise. Lets a caller
        log or quarantine rather than reject.
        """
        if not isinstance(value, str):
            raise MetadataValueError(
                f"{self.name}: expected a string, got {type(value).__name__}"
            )
        lowered = value.lower()
        hits: list[str] = []
        for pattern in self.patterns:
            match = pattern.search(lowered)
            if match:
                hits.append(match.group(0))
        return hits

    def __repr__(self) -> str:
        return (
            f"MetadataSchema(name={self.name!r}, fields={sorted(self.fields)}, "
            f"allow_extra={self.allow_extra})"
        )
