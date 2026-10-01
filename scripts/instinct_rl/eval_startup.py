"""Persist evaluation startup progress even when Isaac Sim exits before recording."""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


class StartupDiagnostics:
    def __init__(self, output):
        self.output = Path(output)
        self.started = time.monotonic()
        self.current_phase = "created"
        self.detail = None
        self.failed = False
        self.returned = False

    def _record(self, status, error=None):
        record = {
            "status": status,
            "phase": self.current_phase,
            "detail": self.detail,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - self.started,
            "error": error,
        }
        # Replace a complete JSON document, so readers never see a partial write.
        path = self.output / "startup_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        message = f"[STAGE] {self.current_phase}: {status} ({record['elapsed_s']:.1f}s)"
        if self.detail:
            message += f" — {self.detail}"
        if error:
            message += f" — {error}"
        with (self.output / "startup.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    def phase(self, name, detail=None):
        self.current_phase, self.detail = name, detail
        self._record("running")

    def failure(self, error):
        """Report before simulator cleanup, which can terminate the interpreter."""
        if self.failed:
            return
        self.failed = True
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        (self.output / "startup_error.txt").write_text(text, encoding="utf-8")
        self._record("error", f"{type(error).__name__}: {error}")
        print(text, file=sys.stderr, flush=True)

    def finish(self):
        if not self.failed and not self.returned:
            self.returned = True
            # summary.json determines whether every planned trial completed.
            self._record("returned")

    def __enter__(self):
        # Fatal native crashes also print Python stacks into stderr / the tee log.
        # This does not catch SIGKILL; the last persisted phase still survives it.
        faulthandler.enable(file=sys.__stderr__)
        self.phase("startup", str(self.output))
        return self

    def __exit__(self, exc_type, error, tb):
        if error is not None:
            self.failure(error)
        else:
            self.finish()
        return False
