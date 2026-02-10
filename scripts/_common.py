"""Shared helpers for the demo scripts."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

# Fix Windows cp1252 encoding — allow Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Base proxy URL (without session prefix)
PROXY_URL = os.environ.get("INTERCEPTOR_URL", "http://127.0.0.1:8080")

WORK_DIR = str(Path(__file__).resolve().parent.parent)

# Resolve system CLI path (the bundled SDK binary may hang on Windows)
CLI_PATH: str | None = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")


def start_session(label: str) -> str:
    """Create a session-tagged proxy URL and set ANTHROPIC_BASE_URL.

    Returns the session ID. The base URL is set to
    http://proxy/_session/{label}-{short_uuid}/  so the proxy can
    extract the session ID and strip the prefix before routing.
    """
    session_id = f"{label}-{uuid.uuid4().hex[:8]}"
    session_url = f"{PROXY_URL}/_session/{session_id}"
    os.environ["ANTHROPIC_BASE_URL"] = session_url
    return session_id


def banner(title: str, session_id: str | None = None) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  Proxy: {PROXY_URL}")
    if session_id:
        print(f"  Session: {session_id}")
    if CLI_PATH:
        print(f"  CLI:   {CLI_PATH}")
    print(f"{'=' * 60}\n")
