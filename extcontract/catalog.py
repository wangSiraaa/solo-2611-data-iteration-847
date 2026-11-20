"""Revision-aware polyline error catalogs built from ingested snapshots.

A catalog answers two questions for a fixed, immutable snapshot of reader rows:

1. For every non-null segment of every channel it stores the exact maximum
   absolute error and earliest attaining time of every pair of same-segment
   endpoints, and can return the minimum point count and a concrete polyline
   for any non-negative error cap (plus the cap chosen at build time).
2. Given a global screen point budget and positive per-channel weights it
   allocates display points so that the maximum of
   ``weight * in-channel max absolute error`` over all channels is minimised,
   subject to every non-empty segment being displayed.

Catalogs are self-contained JSON documents (``polyline-catalog/v1``): they
embed the exact reader snapshot they were derived from, each row's revision,
and the ``sources`` provenance from the ingestion library.  Saving a catalog
and reopening it with :func:`load_catalog` yields an object that answers the
same questions without an ingestion library, so a screen can consume a
reopened catalog directly.  Re-publishing after a correction builds a *new*
catalog; previously saved catalogs remain valid explanations of the older
polylines because nothing mutates them in place.

All times, values, errors, caps and weights are exact rationals; JSON carries
them as minimal rational strings and no float is accepted or produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contract import Contract
from .errors import ContractError
from .ingest import IngestionLibrary
from .polyline import (
    PairError,
    Polyline,
    Sample,
    Segment,
    build_segments,
    channel_curve,
    channel_min_points,
    channel_plan,
    min_polyline,
    pair_error_table,
    polyline_for_count,
)
from .types import DecimalType, IntType, TextType, rational_string

CATALOG_FORMAT = "polyline-catalog/v1"
MAX_POINTS_PER_CHANNEL = 70
MAX_CHANNELS = 12


# ---------------------------------------------------------------------------
# channel selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelSpec:
    """Maps one display channel to three reader ``field_id`` columns.

    The channel name is read from a TEXT column; time and value from numeric
    (INT / DECIMAL) columns.  A null value row splits the channel into two
    segments; a null channel name means the row does not belong to any channel.
    """

    name_field: str
    time_field: str
    value_field: str

    def to_json(self) -> dict:
        return {
            "name_field": self.name_field,
            "time_field": self.time_field,
            "value_field": self.value_field,
        }


# ---------------------------------------------------------------------------
# in-memory catalog model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    time: Fraction
    value: Fraction | None  # None marks a segment break
    row_id: str
    revision: int

    def to_json(self) -> dict:
        return {
            "time": rational_string(self.time),
            "value": None if self.value is None else rational_string(self.value),
            "row_id": self.row_id,
            "revision": self.revision,
        }


@dataclass
class CatalogChannel:
    name: str
    readings: list[Reading]  # strictly increasing times; nulls included as breaks
    segments: list[Segment]
    pair_errors: dict[int, list[PairError]]  # segment index -> all i < j pairs

    def table(self, segment_index: int) -> dict[tuple[int, int], PairError]:
        return {(p.i, p.j): p for p in self.pair_errors[segment_index]}

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "readings": [r.to_json() for r in self.readings],
            "segments": [s.to_json() for s in self.segments],
            "pair_errors": {
                str(index): [p.to_json() for p in pairs]
                for index, pairs in sorted(self.pair_errors.items())
            },
        }


@dataclass
class CatalogBuildResult:
    status: str  # "BUILT" | "REJECTED"
    errors: list[str] = field(default_factory=list)
    catalog: "PolylineCatalog | None" = None

    @property
    def built(self) -> bool:
        return self.status == "BUILT"

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "errors": list(self.errors),
            "catalog": None if self.catalog is None else self.catalog.to_json(),
        }


@dataclass
class PolylineCatalog:
    catalog_id: str
    reader_version: str
    spec: ChannelSpec
    tolerance: Fraction
    channels: list[CatalogChannel]
    snapshot: dict  # the exact reader-records/v1 snapshot this was built from
    fingerprint: str

    # -- queries -----------------------------------------------------------

    def channel(self, name: str) -> CatalogChannel:
        for candidate in self.channels:
            if candidate.name == name:
                return candidate
        raise ContractError(f"unknown channel {name!r}")

    def endpoint_error(
        self, channel_name: str, segment_index: int, i: int, j: int
    ) -> PairError:
        """Exact chord error for endpoints ``i < j`` of one segment."""
        channel = self.channel(channel_name)
        table = channel.table(segment_index)
        key = (i, j)
        if key not in table:
            raise ContractError(
                f"channel {channel_name!r} segment {segment_index}: no endpoint "
                f"pair ({i}, {j})"
            )
        return table[key]

    def earliest_error_time(
        self, channel_name: str, segment_index: int, i: int, j: int
    ) -> Fraction:
        pair = self.endpoint_error(channel_name, segment_index, i, j)
        segment = self._segment(channel_name, segment_index)
        return segment.samples[pair.earliest_index].time

    def minimum_points(
        self, channel_name: str, segment_index: int, cap: Fraction | int | None = None
    ) -> int:
        """Minimum vertices for one segment under ``cap`` (build cap by default)."""
        segment = self._segment(channel_name, segment_index)
        return min_polyline(segment, self.tolerance if cap is None else cap).point_count

    def minimum_polyline(
        self, channel_name: str, segment_index: int, cap: Fraction | int | None = None
    ) -> Polyline:
        segment = self._segment(channel_name, segment_index)
        return min_polyline(segment, self.tolerance if cap is None else cap)

    def build_tolerance_polylines(self, channel_name: str) -> list[Polyline]:
        """One minimum-point polyline per non-empty segment at the build cap."""
        channel = self.channel(channel_name)
        return [min_polyline(segment, self.tolerance) for segment in channel.segments]

    def required_points(self) -> dict[str, int]:
        """Mandatory floor per channel: one point per non-empty segment."""
        return {
            channel.name: channel_min_points(channel.segments)
            for channel in self.channels
        }

    def allocate_display(
        self,
        budget: int,
        weights: dict[str, Fraction | int | str],
    ) -> "DisplayResult":
        """Optimise the weighted worst-channel error under a global point budget.

        Every non-empty segment must be shown.  The objective is the minimum
        attainable value of ``max(weight[c] * channel_error[c])``; all
        comparisons are exact rationals.  When even the mandatory floors do not
        fit, the result is ``INFEASIBLE`` and reports the minimum total needed.
        """
        budget = _require_int(budget, "budget")
        if budget < 0:
            raise ContractError("budget must be a non-negative integer")
        weight_map = self._normalise_weights(weights)

        floors = self.required_points()
        minimum_total = sum(floors.values())
        if budget < minimum_total:
            return DisplayResult(
                status="INFEASIBLE",
                budget=budget,
                minimum_required=minimum_total,
                floors=floors,
                optimal=None,
                channels=[],
                errors=[
                    f"budget {budget} cannot display every non-empty segment; "
                    f"at least {minimum_total} points are required"
                ],
            )

        curves = {
            channel.name: channel_curve(channel.segments) for channel in self.channels
        }
        # Every distinct weighted curve error is a potential optimum; the
        # mandatory-floor boundary (the error at each channel's minimum point
        # count) must be included explicitly, otherwise the binary search could
        # pick an objective below what the floor budget can attain.
        candidates = {Fraction(0)}
        for channel in self.channels:
            weight = weight_map[channel.name]
            # Floor error: one vertex per non-empty segment, so it is the max
            # of the head-to-tail chord errors of the multi-sample segments.
            floor_error = max(
                (
                    channel.table(segment.index)[(0, segment.size - 1)].max_error
                    for segment in channel.segments
                    if segment.size > 1
                ),
                default=Fraction(0),
            )
            candidates.add(weight * floor_error)
            for _count, error in curves[channel.name]:
                candidates.add(weight * error)
        ordered = sorted(candidates)

        def feasible(candidate: Fraction) -> bool:
            total = 0
            for channel in self.channels:
                weight = weight_map[channel.name]
                limit = candidate / weight
                needed = None
                for count, error in curves[channel.name]:
                    if error <= limit:
                        needed = count  # first entry = minimum feasible count
                        break
                if needed is None:  # pragma: no cover - zero-error full count exists
                    return False
                total += needed
            return total <= budget

        lo, hi = 0, len(ordered) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(ordered[mid]):
                hi = mid
            else:
                lo = mid + 1
        optimal = ordered[lo]

        channel_results: list[ChannelDisplay] = []
        for channel in self.channels:
            weight = weight_map[channel.name]
            limit = optimal / weight
            # The minimum feasible count is the first curve entry (points are
            # sorted ascending) whose error fits the weighted limit; unused
            # budget is allowed.
            chosen_count = next(
                count for count, error in curves[channel.name] if error <= limit
            )
            polylines = channel_plan(channel.segments, chosen_count)
            actual_error = max((p.max_error for p in polylines), default=Fraction(0))
            points = [
                _polyline_points(channel, polyline) for polyline in polylines
            ]
            channel_results.append(
                ChannelDisplay(
                    name=channel.name,
                    weight=weight,
                    point_count=chosen_count,
                    max_error=actual_error,
                    weighted_error=weight * actual_error,
                    polylines=polylines,
                    points=points,
                )
            )

        used = sum(channel.point_count for channel in channel_results)
        if used > budget:  # pragma: no cover - binary search guarantees feasibility
            raise ContractError("internal allocation exceeded the budget")

        return DisplayResult(
            status="FEASIBLE",
            budget=budget,
            minimum_required=minimum_total,
            floors=floors,
            optimal=optimal,
            channels=channel_results,
            errors=[],
        )

    # -- (de)serialisation -------------------------------------------------

    def to_json(self) -> dict:
        return {
            "format": CATALOG_FORMAT,
            "catalog_id": self.catalog_id,
            "reader_version": self.reader_version,
            "spec": self.spec.to_json(),
            "tolerance": rational_string(self.tolerance),
            "fingerprint": self.fingerprint,
            "limits": {
                "max_points_per_channel": MAX_POINTS_PER_CHANNEL,
                "max_channels": MAX_CHANNELS,
            },
            "snapshot": self.snapshot,
            "channels": [channel.to_json() for channel in self.channels],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # -- internals ---------------------------------------------------------

    def _segment(self, channel_name: str, segment_index: int) -> Segment:
        channel = self.channel(channel_name)
        for segment in channel.segments:
            if segment.index == segment_index:
                return segment
        raise ContractError(
            f"channel {channel_name!r} has no segment {segment_index}"
        )

    def _normalise_weights(
        self, weights: dict[str, Any]
    ) -> dict[str, Fraction]:
        if not isinstance(weights, dict):
            raise ContractError("weights must map channel name to a positive rational")
        known = {channel.name for channel in self.channels}
        if set(weights) != known:
            missing = sorted(known - set(weights))
            extra = sorted(set(weights) - known)
            problems = []
            if missing:
                problems.append(f"missing weights for {missing}")
            if extra:
                problems.append(f"unknown channels {extra}")
            raise ContractError("; ".join(problems))
        normalised: dict[str, Fraction] = {}
        for name, raw in weights.items():
            weight = _weight_to_fraction(raw)
            if weight <= 0:
                raise ContractError(f"weight for channel {name!r} must be positive")
            normalised[name] = weight
        return normalised


@dataclass
class ChannelDisplay:
    name: str
    weight: Fraction
    point_count: int
    max_error: Fraction
    weighted_error: Fraction
    polylines: list[Polyline]
    points: list[list[Sample]]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "weight": rational_string(self.weight),
            "point_count": self.point_count,
            "max_error": rational_string(self.max_error),
            "weighted_error": rational_string(self.weighted_error),
            "polylines": [p.to_json() for p in self.polylines],
            "points": [[sample.to_json() for sample in run] for run in self.points],
        }


@dataclass
class DisplayResult:
    status: str  # "FEASIBLE" | "INFEASIBLE"
    budget: int
    minimum_required: int
    floors: dict[str, int]
    optimal: Fraction | None
    channels: list[ChannelDisplay]
    errors: list[str]

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "budget": self.budget,
            "minimum_required": self.minimum_required,
            "floors": dict(self.floors),
            "optimal": None if self.optimal is None else rational_string(self.optimal),
            "errors": list(self.errors),
            "channels": [channel.to_json() for channel in self.channels],
        }


# ---------------------------------------------------------------------------
# construction from an ingestion library snapshot
# ---------------------------------------------------------------------------

def build_catalog(
    library: IngestionLibrary,
    spec: ChannelSpec,
    *,
    tolerance: Fraction | int = 0,
    catalog_id: str | None = None,
    reader_version: str | None = None,
) -> CatalogBuildResult:
    """Build a catalog from the library's current target snapshot.

    The snapshot is captured once and embedded in the catalog; later
    corrections cannot mutate a returned catalog.  Data problems (unknown
    fields, non-numeric time/value, duplicate times for one channel, or scale
    limit breaches) produce a ``REJECTED`` result with every collected error.
    """
    errors: list[str] = []
    tolerance = _validate_tolerance(tolerance)
    _validate_spec_against_contract(library.reader, spec, errors)
    if errors:
        return CatalogBuildResult("REJECTED", errors, None)

    snapshot = library.publish(reader_version)
    rows = snapshot.get("rows", [])
    grouped: dict[str, list[Reading]] = {}
    order: list[str] = []

    for row in rows:
        row_id = row["row_id"]
        revision = row["revision"]
        values = row["values"]
        name = values.get(spec.name_field)
        if name is None:
            # A null channel name means the row is not part of any channel.
            continue
        if not isinstance(name, str):
            errors.append(
                f"row {row_id!r}: channel name field {spec.name_field!r} must be TEXT"
            )
            continue
        if spec.time_field not in values:
            errors.append(
                f"row {row_id!r}: missing time field {spec.time_field!r}"
            )
            continue
        raw_time = values[spec.time_field]
        if raw_time is None:
            errors.append(
                f"row {row_id!r}: time field {spec.time_field!r} may not be null"
            )
            continue
        time_value, time_ok = _numeric_value(raw_time, spec.time_field, row_id)
        if not time_ok:
            errors.append(time_value)
            continue
        if spec.value_field not in values:
            errors.append(
                f"row {row_id!r}: missing value field {spec.value_field!r}"
            )
            continue
        raw_value = values[spec.value_field]
        if raw_value is None:
            value = None
        else:
            value, value_ok = _numeric_value(raw_value, spec.value_field, row_id)
            if not value_ok:
                errors.append(value)
                continue
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(
            Reading(time=time_value, value=value, row_id=row_id, revision=revision)
        )

    if len(order) > MAX_CHANNELS:
        errors.append(
            f"channel count {len(order)} exceeds the {MAX_CHANNELS} channel limit"
        )

    channels: list[CatalogChannel] = []
    for name in order:
        readings = grouped[name]
        readings.sort(key=lambda reading: reading.time)
        seen_times: set[Fraction] = set()
        for reading in readings:
            if reading.time in seen_times:
                errors.append(
                    f"channel {name!r}: duplicate time "
                    f"{rational_string(reading.time)} from row {reading.row_id!r}; "
                    f"valid times of one channel must be unique"
                )
            seen_times.add(reading.time)
        non_null = sum(1 for reading in readings if reading.value is not None)
        if non_null > MAX_POINTS_PER_CHANNEL:
            errors.append(
                f"channel {name!r}: {non_null} non-null points exceed the "
                f"{MAX_POINTS_PER_CHANNEL}-point per-channel limit"
            )
        if errors:
            continue
        segments = build_segments(
            (reading.time, reading.value) for reading in readings
        )
        pair_errors = {
            segment.index: list(pair_error_table(segment.samples).values())
            for segment in segments
        }
        channels.append(
            CatalogChannel(
                name=name,
                readings=readings,
                segments=segments,
                pair_errors=pair_errors,
            )
        )

    if errors:
        return CatalogBuildResult("REJECTED", errors, None)

    fingerprint = _catalog_fingerprint(snapshot, spec, tolerance, channels)
    catalog = PolylineCatalog(
        catalog_id=catalog_id or f"catalog-{fingerprint[:12]}",
        reader_version=snapshot["reader_version"],
        spec=spec,
        tolerance=tolerance,
        channels=channels,
        snapshot=snapshot,
        fingerprint=fingerprint,
    )
    return CatalogBuildResult("BUILT", [], catalog)


# ---------------------------------------------------------------------------
# loading a saved catalog (no library needed)
# ---------------------------------------------------------------------------

def load_catalog(document: Any) -> PolylineCatalog:
    """Reopen a ``polyline-catalog/v1`` document for direct screen consumption."""
    if not isinstance(document, dict):
        raise ContractError("catalog document must be a JSON object")
    if document.get("format") != CATALOG_FORMAT:
        raise ContractError(
            f"expected format {CATALOG_FORMAT!r}, got {document.get('format')!r}"
        )
    required = {
        "catalog_id",
        "reader_version",
        "spec",
        "tolerance",
        "fingerprint",
        "snapshot",
        "channels",
    }
    missing = required - set(document)
    if missing:
        raise ContractError(f"catalog missing keys {sorted(missing)}")

    spec_raw = document["spec"]
    spec = ChannelSpec(
        name_field=spec_raw["name_field"],
        time_field=spec_raw["time_field"],
        value_field=spec_raw["value_field"],
    )
    tolerance = Fraction(document["tolerance"])
    if tolerance < 0:
        raise ContractError("catalog tolerance must be non-negative")

    channels: list[CatalogChannel] = []
    for channel_raw in document["channels"]:
        readings = [
            Reading(
                time=Fraction(item["time"]),
                value=None if item["value"] is None else Fraction(item["value"]),
                row_id=item["row_id"],
                revision=item["revision"],
            )
            for item in channel_raw["readings"]
        ]
        segments: list[Segment] = []
        for segment_raw in channel_raw["segments"]:
            samples = tuple(
                Sample(time=Fraction(item["time"]), value=Fraction(item["value"]))
                for item in segment_raw["samples"]
            )
            segments.append(Segment(index=segment_raw["index"], samples=samples))
        pair_errors: dict[int, list[PairError]] = {}
        for index_raw, pairs in channel_raw["pair_errors"].items():
            pair_errors[int(index_raw)] = [
                PairError(
                    i=item["i"],
                    j=item["j"],
                    max_error=Fraction(item["max_error"]),
                    earliest_index=item["earliest_index"],
                )
                for item in pairs
            ]
        channels.append(
            CatalogChannel(
                name=channel_raw["name"],
                readings=readings,
                segments=segments,
                pair_errors=pair_errors,
            )
        )

    catalog = PolylineCatalog(
        catalog_id=document["catalog_id"],
        reader_version=document["reader_version"],
        spec=spec,
        tolerance=tolerance,
        channels=channels,
        snapshot=document["snapshot"],
        fingerprint=document["fingerprint"],
    )
    # Re-derived content must match the saved fingerprint: a tampered or
    # truncated catalog is refused rather than silently reinterpreted.
    expected = _catalog_fingerprint(
        document["snapshot"], spec, tolerance, channels
    )
    if expected != catalog.fingerprint:
        raise ContractError("catalog fingerprint does not match its embedded snapshot")
    return catalog


def load_catalog_file(path: str | Path) -> PolylineCatalog:
    return load_catalog(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# validation / conversion helpers
# ---------------------------------------------------------------------------

def _validate_spec_against_contract(
    reader: Contract, spec: ChannelSpec, errors: list[str]
) -> None:
    if not isinstance(spec, ChannelSpec):
        errors.append("channel spec must be a ChannelSpec")
        return
    fields = reader.field_by_id()
    for role_name, field_id in (
        ("name", spec.name_field),
        ("time", spec.time_field),
        ("value", spec.value_field),
    ):
        if field_id not in fields:
            errors.append(
                f"{role_name} field {field_id!r} is not declared by reader "
                f"{reader.version!r}"
            )
    if errors:
        return
    if not isinstance(fields[spec.name_field].type, TextType):
        errors.append(
            f"name field {spec.name_field!r} must be TEXT, got "
            f"{fields[spec.name_field].type.describe()}"
        )
    for role_name, field_id in (("time", spec.time_field), ("value", spec.value_field)):
        ftype = fields[field_id].type
        if not isinstance(ftype, (IntType, DecimalType)):
            errors.append(
                f"{role_name} field {field_id!r} must be INT or DECIMAL, got "
                f"{ftype.describe()} (TEXT is never parsed as a number)"
            )


def _numeric_value(raw: Any, field_id: str, row_id: str) -> tuple[Any, bool]:
    """Decode an encoded reader rational.

    Returns ``(Fraction, True)`` on success or ``(error_message, False)``; the
    caller owns appending the message so control flow never depends on parsing
    a previously formatted string.
    """
    if not isinstance(raw, str):
        return (
            f"row {row_id!r}: field {field_id!r} must encode a rational as a "
            f"minimal string, got {type(raw).__name__}",
            False,
        )
    try:
        return Fraction(raw), True
    except (ValueError, ZeroDivisionError):
        return (
            f"row {row_id!r}: field {field_id!r} has invalid rational {raw!r}",
            False,
        )


def _validate_tolerance(tolerance: Any) -> Fraction:
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, Fraction)):
        raise ContractError("tolerance must be a non-negative exact rational")
    tolerance = Fraction(tolerance)
    if tolerance < 0:
        raise ContractError("tolerance must be non-negative")
    return tolerance


def _weight_to_fraction(raw: Any) -> Fraction:
    if isinstance(raw, bool):
        raise ContractError("weights must be positive rationals, not booleans")
    if isinstance(raw, (int, Fraction)):
        return Fraction(raw)
    if isinstance(raw, str):
        try:
            return Fraction(raw)
        except (ValueError, ZeroDivisionError):
            pass
    raise ContractError(
        f"weight {raw!r} must be a positive int, Fraction, or exact fraction string"
    )


def _require_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ContractError(f"{where} must be an integer")
    return value


def _polyline_points(channel: CatalogChannel, polyline: Polyline) -> list[Sample]:
    segment = next(s for s in channel.segments if s.index == polyline.segment_index)
    return [segment.samples[index] for index in polyline.vertex_indices]


def _catalog_fingerprint(
    snapshot: dict,
    spec: ChannelSpec,
    tolerance: Fraction,
    channels: Sequence[CatalogChannel] | None = None,
) -> str:
    """Content hash over the snapshot, selection, cap, and all derived geometry.

    Channels are serialised through their canonical JSON form so any edit to a
    reading, segment split, or stored pair error changes the hash; reopening a
    mismatched document is refused.
    """
    payload = json.dumps(
        {
            "snapshot": snapshot,
            "spec": spec.to_json(),
            "tolerance": rational_string(tolerance),
            "channels": (
                [channel.to_json() for channel in channels]
                if channels is not None
                else []
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
