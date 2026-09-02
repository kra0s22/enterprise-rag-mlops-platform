"""Tenant scoping helpers for the API layer.

Every chunk is tagged with a ``tenant_id`` at ingestion; retrieval must therefore
be scoped to one tenant. ``scope_filters`` injects the tenant into the payload
filters before they reach the vector store, so a request can only see the corpus
of its own tenant on a shared collection.
"""

from __future__ import annotations

from typing import Any


def scope_filters(
    filters: dict[str, Any],
    tenant_id: str | None,
    default_tenant: str,
) -> dict[str, Any]:
    """Return the request filters with ``tenant_id`` enforced.

    The effective tenant is the request value (when provided) or the platform
    default; it cannot be absent, so unauthenticated or misconfigured clients
    never fall back to a cross-tenant scan.
    """
    return {**filters, "tenant_id": tenant_id or default_tenant}
