"""Small structured metric and timing sinks used by harness orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol, TextIO

type MetricValue = int | float
type MetricFields = Mapping[str, str | int | float | bool | None]


def _checked_name(name: str) -> str:
    if not name or any(character.isspace() for character in name):
        raise ValueError("metric names must be non-empty and contain no whitespace")
    return name


def _checked_value(value: MetricValue) -> MetricValue:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("metric values must be numeric")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("metric values must be finite")
    return value


def _checked_fields(fields: MetricFields) -> dict[str, str | int | float | bool | None]:
    checked: dict[str, str | int | float | bool | None] = {}
    for key, value in fields.items():
        if not key:
            raise ValueError("metric field names cannot be empty")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("metric field values must be finite")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise TypeError("metric field values must be JSON scalars")
        checked[key] = value
    return checked


class MetricSink(Protocol):
    """Receive a named finite scalar and JSON-scalar dimensions."""

    def record(
        self, name: str, value: MetricValue, /, **fields: str | int | float | bool | None
    ) -> None:
        """Write one metric event."""


class TimerSink(Protocol):
    """Create a scope that records elapsed monotonic time as a metric."""

    def timer(
        self, name: str, /, **fields: str | int | float | bool | None
    ) -> AbstractContextManager[None]:
        """Return a context manager for one timed scope."""


class NoOpMetricSink:
    """The default zero-allocation metric destination."""

    def record(
        self, name: str, value: MetricValue, /, **fields: str | int | float | bool | None
    ) -> None:
        _checked_name(name)
        _checked_value(value)
        _checked_fields(fields)


class NoOpTimerSink:
    """A no-op timing destination that preserves the timing call shape."""

    @contextmanager
    def timer(self, name: str, /, **fields: str | int | float | bool | None) -> Iterator[None]:
        _checked_name(name)
        _checked_fields(fields)
        yield


@dataclass(slots=True)
class JsonlMetricSink:
    """Append deterministic structured events to a UTF-8 JSON Lines stream."""

    _stream: TextIO
    _owns_stream: bool = False

    @classmethod
    def open(cls, path: Path) -> JsonlMetricSink:
        """Create parent directories and open a line-buffered JSONL sink."""

        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path.open("a", encoding="utf-8", buffering=1), True)

    def close(self) -> None:
        """Close a stream opened by :meth:`open`; injected streams remain caller-owned."""

        if self._owns_stream:
            self._stream.close()
            self._owns_stream = False

    def __enter__(self) -> JsonlMetricSink:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def record(
        self, name: str, value: MetricValue, /, **fields: str | int | float | bool | None
    ) -> None:
        payload: dict[str, object] = {
            "kind": "metric",
            "name": _checked_name(name),
            "value": _checked_value(value),
            "fields": _checked_fields(fields),
        }
        self._stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()

    @contextmanager
    def timer(self, name: str, /, **fields: str | int | float | bool | None) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.record(name, perf_counter() - started, unit="seconds", **fields)


JSONLMetricSink = JsonlMetricSink
