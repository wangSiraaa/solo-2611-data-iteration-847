"""Exact polyline compression over rational time/value samples.

Every quantity here is an exact :class:`~fractions.Fraction` (or ``int``);
no floating-point arithmetic is ever used, so two endpoints that are equal as
rationals compare equal without any tolerance.  The module is deliberately
pure: it knows nothing about ingestion, revisions, or JSON documents.

Segment model
-------------
A *segment* is a run of non-null ``(time, value)`` samples at strictly
increasing times.  A null sample in the source channel breaks the run; lines
are never drawn across that gap.

  * A segment with zero samples is not displayed at all.
  * A one-sample segment keeps that single point (a degenerate polyline with
    one vertex, no edges, zero error).
  * A longer segment is compressed by keeping original samples, in increasing
    time order, always including the segment head and tail.  Between two kept
    endpoints the display draws a straight line.

Error of a chord
----------------
For endpoint pair ``(i, j)`` (``i < j``) the display line interpolates every
clamped original sample ``k`` with ``i <= k <= j``::

    line(t_k) = v_i + (v_j - v_i) * (t_k - t_i) / (t_j - t_i)

and the chord error is the maximum of ``|v_k - line(t_k)|`` over the clamped
samples.  Endpoints themselves contribute exactly zero, so the answer is
determined by interior samples.  :func:`pair_error` also reports the earliest
sample index attaining that maximum.

Exact minimisation
------------------
For a given non-negative error cap :math:`\\epsilon`, :func:`min_polyline`
returns the minimum number of vertices and a concrete vertex-index polyline
attaining it.  The optimum is found by exact dynamic programming (greedy
farthest-reach rules are not trusted here); ties choose the polyline whose
reconstruction picks the earliest reachable vertices, which is deterministic.

:func:`segment_curve` and :func:`channel_curve` expose the full
"vertex count -> minimum achievable maximum error" trade-off curve used by the
screen budget allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from .errors import ContractError

Rational = int | Fraction


@dataclass(frozen=True)
class Sample:
    """One exact ``(time, value)`` reading."""

    time: Fraction
    value: Fraction

    def to_json(self) -> dict:
        return {"time": str(self.time), "value": str(self.value)}


@dataclass(frozen=True)
class PairError:
    """Maximum absolute error of the chord between two segment endpoints."""

    i: int
    j: int
    max_error: Fraction
    earliest_index: int  # earliest k in [i, j] attaining max_error

    def to_json(self) -> dict:
        return {
            "i": self.i,
            "j": self.j,
            "max_error": str(self.max_error),
            "earliest_index": self.earliest_index,
        }


@dataclass(frozen=True)
class Segment:
    """One non-null run: samples at strictly increasing times."""

    index: int
    samples: tuple[Sample, ...]

    @property
    def size(self) -> int:
        return len(self.samples)

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "samples": [s.to_json() for s in self.samples],
        }


@dataclass(frozen=True)
class Polyline:
    """A concrete polyline: vertex indices into the owning segment's samples."""

    segment_index: int
    vertex_indices: tuple[int, ...]
    max_error: Fraction

    @property
    def point_count(self) -> int:
        return len(self.vertex_indices)

    def to_json(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "vertex_indices": list(self.vertex_indices),
            "max_error": str(self.max_error),
        }


# ---------------------------------------------------------------------------
# segment construction from raw (time, value) readings, nulls break segments
# ---------------------------------------------------------------------------

def build_segments(readings: Iterable[tuple[Rational, Rational | None]]) -> list[Segment]:
    """Split ``readings`` into non-null segments.

    ``readings`` must already be sorted by time with unique times (the catalog
    builder enforces both).  A ``None`` value closes the current segment; no
    line ever spans the resulting gap.  Empty runs are omitted, single-sample
    runs are kept.
    """
    segments: list[Segment] = []
    run: list[Sample] = []

    def flush() -> None:
        if run:
            segments.append(Segment(index=len(segments), samples=tuple(run)))
            run.clear()

    for time, value in readings:
        t = _as_fraction(time, "time")
        if value is None:
            flush()
            continue
        run.append(Sample(time=t, value=_as_fraction(value, "value")))
    flush()
    return segments


# ---------------------------------------------------------------------------
# exact chord geometry
# ---------------------------------------------------------------------------

def interpolate(samples: Sequence[Sample], i: int, j: int, k: int) -> Fraction:
    """Exact value at ``t_k`` of the straight line through samples ``i`` and ``j``."""
    if i == j:
        return samples[i].value
    a, b, p = samples[i], samples[j], samples[k]
    span = b.time - a.time
    if span == 0:  # unique times make this unreachable; keep the guard explicit
        raise ContractError("cannot interpolate between two equal times")
    return a.value + (b.value - a.value) * (p.time - a.time) / span


def pair_error(samples: Sequence[Sample], i: int, j: int) -> PairError:
    """Max absolute interpolation error of chord ``(i, j)`` over clamped samples.

    The earliest sample attaining the maximum is recorded.  Endpoints always
    have zero error, so for adjacent pairs (and any perfectly collinear chord)
    the answer is zero at index ``i``.
    """
    if not (0 <= i < j < len(samples)):
        raise ContractError(f"pair_error requires 0 <= i < j < n, got i={i}, j={j}")
    best = Fraction(0)
    earliest = i
    for k in range(i, j + 1):
        deviation = abs(samples[k].value - interpolate(samples, i, j, k))
        if deviation > best:
            best = deviation
            earliest = k
    return PairError(i=i, j=j, max_error=best, earliest_index=earliest)


def pair_error_table(samples: Sequence[Sample]) -> dict[tuple[int, int], PairError]:
    """All ``i < j`` chord errors for one segment (exact)."""
    table: dict[tuple[int, int], PairError] = {}
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            table[(i, j)] = pair_error(samples, i, j)
    return table


def _polyline_error(
    vertices: Sequence[int], table: dict[tuple[int, int], PairError]
) -> Fraction:
    error = Fraction(0)
    for a, b in zip(vertices, vertices[1:]):
        entry = table[(a, b)]
        if entry.max_error > error:
            error = entry.max_error
    return error


# ---------------------------------------------------------------------------
# exact per-segment minimisation
# ---------------------------------------------------------------------------

def min_polyline(
    segment: Segment, cap: Rational
) -> Polyline:
    """Minimum-vertex polyline whose every chord error is ``<= cap``.

    The segment head and tail are always vertices.  An exact DP over vertex
    counts returns both the minimum count and a concrete, deterministic
    polyline (earliest reachable vertices win ties).  A one-sample segment
    yields its single point with zero error.
    """
    cap = _as_fraction(cap, "cap")
    if cap < 0:
        raise ContractError("error cap must be non-negative")
    n = segment.size
    if n == 0:
        raise ContractError("empty segments are never displayed")
    if n == 1:
        return Polyline(segment.index, (0,), Fraction(0))

    table = pair_error_table(segment.samples)
    curve = _segment_dp(n, table)
    entry = _best_entry_for_cap(curve, cap)
    vertices = _reconstruct(n, table, target_count=entry[0])
    return Polyline(segment.index, tuple(vertices), _polyline_error(vertices, table))


def segment_curve(segment: Segment) -> list[tuple[int, Fraction, tuple[int, ...]]]:
    """Trade-off curve: ``(point_count, min_max_error, vertex_indices)``.

    One entry per feasible vertex count from 2 (or 1 for a single sample) to
    the full sample count; counts are increasing and errors are non-increasing.
    The full-count polyline has zero error because every chord is adjacent.
    """
    n = segment.size
    if n == 0:
        return []
    if n == 1:
        return [(1, Fraction(0), (0,))]
    table = pair_error_table(segment.samples)
    curve = _segment_dp(n, table)
    result = []
    for count in sorted(curve):
        vertices = tuple(_reconstruct(n, table, target_count=count))
        result.append((count, curve[count], vertices))
    return result


def _segment_dp(
    n: int, table: dict[tuple[int, int], PairError]
) -> dict[int, Fraction]:
    """Minimum achievable max error for each kept-point count of the segment."""
    errors, _parents = _segment_dp_with_parents(n, table)
    return errors


def _segment_dp_with_parents(
    n: int, table: dict[tuple[int, int], PairError]
) -> tuple[dict[int, Fraction], dict[tuple[int, int], int]]:
    """Exact DP returning both optimum errors and predecessor pointers.

    A multi-sample segment's floor is two points: head and tail joined by one
    chord with error ``table[(0, n-1)]``.  Counts 2..n are the DP optima; the
    full count has zero error.  A single-sample segment has one option, its
    lone point.  The parent map records, for ``(count, end_index)``, the
    predecessor index used by an optimal polyline; ties keep the earliest
    predecessor so reconstruction is deterministic.
    """
    if n == 1:
        return {1: Fraction(0)}, {}
    errors: dict[int, Fraction] = {2: table[(0, n - 1)].max_error}
    # dp[c][j] = minimum max error of a c-point polyline from 0 ending at j.
    dp: list[dict[int, Fraction]] = [{} for _ in range(n + 1)]
    parent: dict[tuple[int, int], int] = {}
    dp[1][0] = Fraction(0)
    for c in range(2, n + 1):
        for j in range(1, n):
            best: Fraction | None = None
            best_i: int | None = None
            for i in range(0, j):
                prior = dp[c - 1].get(i)
                if prior is None:
                    continue
                candidate = max(prior, table[(i, j)].max_error)
                # Iterate i ascending, so a strict comparison keeps the
                # earliest predecessor on equal errors.
                if best is None or candidate < best:
                    best = candidate
                    best_i = i
            if best is not None:
                dp[c][j] = best
                parent[(c, j)] = best_i
    for c in range(2, n + 1):
        if n - 1 in dp[c]:
            errors[c] = dp[c][n - 1]
    return errors, parent


def _best_entry_for_cap(curve: dict[int, Fraction], cap: Fraction) -> tuple[int, Fraction]:
    feasible = [(c, e) for c, e in curve.items() if e <= cap]
    if not feasible:
        # Only reachable when cap < 0, already rejected above; kept for safety.
        raise ContractError("no polyline satisfies the requested error cap")
    count, error = min(feasible, key=lambda item: item[0])
    return count, error


def _reconstruct(
    n: int,
    table: dict[tuple[int, int], PairError],
    target_count: int,
) -> list[int]:
    """Backtrack the DP parent chain for an exact ``target_count``-point line.

    Following stored predecessors guarantees the result attains the DP optimum
    (no greedy reachability assumption); earliest-predecessor ties make the
    returned vertex set deterministic.
    """
    if n == 1:
        if target_count != 1:  # pragma: no cover - caller range-checks
            raise ContractError("internal polyline reconstruction failed")
        return [0]
    _errors, parent = _segment_dp_with_parents(n, table)
    vertices = [n - 1]
    count = target_count
    end = n - 1
    while count > 1:
        predecessor = parent.get((count, end))
        if predecessor is None:  # pragma: no cover - DP guarantees the chain
            raise ContractError("internal polyline reconstruction failed")
        vertices.append(predecessor)
        end = predecessor
        count -= 1
    if end != 0:  # pragma: no cover - every chain starts at the head
        raise ContractError("internal polyline reconstruction failed")
    vertices.reverse()
    return vertices


# ---------------------------------------------------------------------------
# channel-level curve: combine independent segments
# ---------------------------------------------------------------------------

def polyline_for_count(segment: Segment, count: int) -> Polyline:
    """Deterministic polyline using exactly ``count`` vertices.

    Its error is the segment curve's optimum for that count; ties keep the
    earliest reachable vertices.  ``count`` ranges from 1 (single-sample
    segments only) to the full sample count (zero error).
    """
    n = segment.size
    if not (1 <= count <= n):
        raise ContractError(f"vertex count {count} outside [1, {n}]")
    if n == 1:
        if count != 1:  # pragma: no cover - range check above handles it
            raise ContractError("a one-sample segment only supports one vertex")
        return Polyline(segment.index, (0,), Fraction(0))
    table = pair_error_table(segment.samples)
    curve = _segment_dp(n, table)
    if count not in curve:  # pragma: no cover - range check covers the bounds
        raise ContractError(f"vertex count {count} is not attainable")
    vertices = _reconstruct(n, table, target_count=count)
    return Polyline(segment.index, tuple(vertices), _polyline_error(vertices, table))


def channel_plan(
    segments: Sequence[Segment], total_points: int
) -> list[Polyline]:
    """Distribute exactly ``total_points`` across segments at minimum max error.

    Every non-empty segment receives at least one vertex; the exact DP also
    records the chosen split so a concrete, deterministic polyline per segment
    is returned.  Null gaps are never bridged because segments stay separate.
    """
    floor = channel_min_points(segments)
    if total_points < floor:
        raise ContractError(
            f"channel needs at least {floor} points, got {total_points}"
        )
    options: list[list[tuple[int, Fraction, tuple[int, ...]]]] = []
    for segment in segments:
        if segment.size == 1:
            options.append([(1, Fraction(0), (0,))])
        else:
            options.append([(c, e, verts) for c, e, verts in segment_curve(segment)])

    # state: total count -> (best error, real parent count, chosen option).
    # The initial anchor is count 0; parents always store a real count from
    # the previous stage (0 for the first segment), so backtracking is correct
    # even when single-sample segments sit before longer ones.
    state: dict[int, tuple[Fraction, int, int]] = {0: (Fraction(0), -1, -1)}
    parents: list[dict[int, tuple[Fraction, int, int]]] = []
    for seg_options in options:
        nxt: dict[int, tuple[Fraction, int, int]] = {}
        for count, (error, _prev, _opt) in state.items():
            for option_index, (added, seg_error, _verts) in enumerate(seg_options):
                total = count + added
                candidate = max(error, seg_error)
                if total not in nxt or candidate < nxt[total][0]:
                    nxt[total] = (candidate, count, option_index)
        parents.append(nxt)
        state = nxt

    if total_points not in state:
        raise ContractError(
            f"point count {total_points} is not attainable for this channel"
        )

    chosen_options: list[int] = []
    cursor = total_points
    for seg_index in range(len(options) - 1, -1, -1):
        _error, previous, option_index = parents[seg_index][cursor]
        chosen_options.append(option_index)
        cursor = previous
    chosen_options.reverse()

    plan: list[Polyline] = []
    for segment, option_index in zip(segments, chosen_options):
        count = options[segment.index][option_index][0]
        plan.append(polyline_for_count(segment, count))
    if sum(p.point_count for p in plan) != total_points:  # pragma: no cover
        raise ContractError("internal channel plan failed")
    return plan


def channel_curve(
    segments: Sequence[Segment],
) -> list[tuple[int, Fraction]]:
    """``(total_point_count, min_max_error)`` curve for one whole channel.

    Segments are independent (null gaps are never bridged), so their trade-off
    curves are combined by exact dynamic programming: for every total point
    count keep the smallest possible worst segment error.  Single-point
    segments contribute their mandatory one point and zero error.
    """
    # state: total points -> minimum max error.  Each segment contributes its
    # own independent option list (single-sample segments have exactly one:
    # their mandatory point at zero error), so the order of segments and the
    # presence of null gaps never change the result.
    state: dict[int, Fraction] = {0: Fraction(0)}
    for segment in segments:
        if segment.size == 1:
            seg_options = [(1, Fraction(0))]
        else:
            seg_options = list(
                _segment_dp(segment.size, pair_error_table(segment.samples)).items()
            )
        nxt: dict[int, Fraction] = {}
        for count, error in state.items():
            for added, seg_error in seg_options:
                total = count + added
                candidate = max(error, seg_error)
                if total not in nxt or candidate < nxt[total]:
                    nxt[total] = candidate
        state = nxt
    return sorted(state.items())


def channel_min_points(segments: Sequence[Segment]) -> int:
    """Mandatory display floor for one channel.

    Every non-empty segment must appear: a single-sample segment keeps its one
    point, a longer segment keeps at least its head and tail (two points).
    Null gaps are never bridged because segments are counted independently.
    """
    return sum(1 if segment.size == 1 else 2 for segment in segments)


def channel_error_for_budget(
    segments: Sequence[Segment], point_budget: int
) -> Fraction:
    """Minimum achievable in-channel max error using at most ``point_budget``."""
    floor = channel_min_points(segments)
    if point_budget < floor:
        raise ContractError(
            f"channel needs at least {floor} points, got budget {point_budget}"
        )
    curve = channel_curve(segments)
    affordable = [(count, error) for count, error in curve if count <= point_budget]
    # The curve lists increasing counts; the last affordable entry is optimal.
    return affordable[-1][1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_fraction(value: Rational, where: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ContractError(
            f"{where} must be an exact int/Fraction, got {type(value).__name__}"
        )
    return Fraction(value)
