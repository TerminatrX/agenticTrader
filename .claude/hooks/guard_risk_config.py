"""PreToolUse guard: the agent may not edit risk limits or bless its own edits.

Reads the hook payload on stdin and emits a `deny` decision when the proposed
tool call would either:

  1. modify `config/risk.yaml` or `config/risk.lock`, or
  2. run `lock-risk`, the command that re-approves changed risk values.

Both matter. Blocking (1) alone would leave the agent able to edit the file and
then re-lock it, which is the same as no protection at all.

This is the harness half of a two-layer control. It constrains *this agent*.
The other half — the SHA-256 baseline in `config/risk.lock`, verified on every
config load — catches edits from any source, including ones this hook never
sees. Neither layer is meant to stop a human; both exist to stop an agent from
quietly widening a limit to fit a trade that was correctly blocked.

Exit code is always 0. The decision travels in the JSON on stdout, so a missing
or broken hook cannot brick the session by accidentally exiting non-zero.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import PurePath

PROTECTED_FILENAMES = {"risk.yaml", "risk.lock"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
FORBIDDEN_COMMAND = "lock-risk"

# Shell commands that would WRITE a protected file. Reading one (`cat`,
# `Get-Content`) stays allowed — inspecting the limits is normal and useful;
# rewriting them is not.
#
# The write must actually target the protected file. An earlier version merely
# asked whether the command contained a write verb *and* mentioned the file
# anywhere, which denied `pytest 2>&1 | grep risk.lock` — the `2>&1` counted as
# a redirect. A guard that blocks ordinary work gets disabled, so precision
# here is a safety property, not a nicety.
#
# This remains a speed bump, not a boundary. A heredoc or a Python one-liner
# still gets through. The boundary is the SHA-256 baseline in config/risk.lock,
# verified on every config load, which catches a change however it was made.
_PROTECTED_RE = r"risk\.(?:yaml|lock)"

SHELL_WRITE_PATTERNS = (
    # Redirection whose target is the protected file: `> config/risk.yaml`.
    # `&` and `|` are excluded from the path so `2>&1` cannot match.
    re.compile(r">>?\s*[\"']?[^\s\"'|;&]*" + _PROTECTED_RE, re.IGNORECASE),
    # A write tool naming the file later in the same command segment, which
    # covers destination-last forms like `cp src config/risk.lock`.
    re.compile(
        r"\b(?:tee|sed\s+-i|set-content|out-file|add-content|cp|mv|copy|move"
        r"|truncate|dd|install|rm|del|remove-item)\b[^;|&]*" + _PROTECTED_RE,
        re.IGNORECASE,
    ),
)

DENY_EDIT = (
    "BLOCKED: {name} defines the risk limits this system trades under, and the "
    "agent may not change them. If a limit genuinely needs to move, the human "
    "edits config/risk.yaml directly and then runs "
    "`python -m agentic_trader.cli lock-risk --confirm`. Never loosen a limit "
    "to make a blocked trade fit — a rejection is the system working."
)
DENY_LOCK = (
    "BLOCKED: `lock-risk` re-approves the current risk values as the trusted "
    "baseline. It is a human command by design; an agent that could run it "
    "could edit a limit and immediately bless the edit."
)
DENY_SHELL_WRITE = (
    "BLOCKED: this command appears to write to a protected risk-config file. "
    "Reading them is fine; changing them is the human's call. If a limit needs "
    "to move, the human edits config/risk.yaml and runs "
    "`python -m agentic_trader.cli lock-risk --confirm`."
)


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def targets_protected_file(tool_input: dict) -> str | None:
    """Return the protected filename this call would write, if any.

    Matches on basename rather than full path: path forms vary (absolute,
    relative, Windows separators, symlinks), and a guard that can be sidestepped
    by spelling the path differently is not a guard.
    """
    for key in ("file_path", "notebook_path", "path"):
        raw = tool_input.get(key)
        if not raw:
            continue
        name = PurePath(str(raw).replace("\\", "/")).name
        if name in PROTECTED_FILENAMES:
            return name
    return None


def main() -> None:
    raw = sys.stdin.read()

    try:
        payload = json.loads(raw)
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}

        if tool_name in EDIT_TOOLS:
            hit = targets_protected_file(tool_input)
            if hit:
                deny(DENY_EDIT.format(name=hit))

        elif tool_name in SHELL_TOOLS:
            command = str(tool_input.get("command", ""))
            lowered = command.lower()

            if FORBIDDEN_COMMAND in lowered:
                deny(DENY_LOCK)

            if any(p.search(command) for p in SHELL_WRITE_PATTERNS):
                deny(DENY_SHELL_WRITE)

    except Exception:
        # Fail closed, but only narrowly. Denying everything on a parse bug
        # would brick the session; denying nothing would silently disable the
        # guard. So fall back to a substring check on the raw payload and deny
        # only when it plausibly touches something protected.
        haystack = raw.lower()
        if any(t in haystack for t in ("risk.yaml", "risk.lock", "lock-risk")):
            deny(
                "BLOCKED: the risk-config guard could not parse this tool call and "
                "the payload references a protected file. Refusing by default."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
