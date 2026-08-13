"""Agentic trader: deterministic trading core driven by an agentic orchestrator.

Architectural contract — read this before adding a module:

    Nothing in this package performs I/O against a broker.

Market data arrives, and orders depart, exclusively through the Robinhood MCP
tools invoked by the orchestrating agent. Python's job is to turn raw payloads
into decisions, deterministically and reproducibly, so that every decision can
be replayed and audited offline.

Concretely, that means no module here may import `requests`, `httpx`, or any
MCP client. Functions take data in and return decisions out. The only side
effects permitted are writes to the local journal and audit log.
"""

__version__ = "0.1.0"
