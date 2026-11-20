"""Exception types used by the external contract toolkit.

The public services mostly return structured result dictionaries; these
exceptions are reserved for malformed caller input and for internal
"should never happen" invariants.  Business rejections (incompatible
contracts, revision conflicts, domain violations in a file) are returned
as explicit result objects, never raised, so that a caller cannot mistake
a control-flow accident for an acceptance.
"""


class ExtContractError(Exception):
    """Base class for all toolkit errors."""


class ContractError(ExtContractError):
    """A contract document is structurally invalid."""


class DomainError(ExtContractError, ValueError):
    """A value is not a legal member of a declared type domain."""


class ConversionError(ExtContractError, ValueError):
    """Exact conversion failed (must never happen for a SAFE plan)."""


class LimitError(ExtContractError):
    """A documented scale limit was exceeded."""


class IngestRejected(ExtContractError):
    """Raised only by the ``*_or_raise`` helper; normal API returns results."""
