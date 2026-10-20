"""
src/flow_stream.py
==================
Incremental CSV reader for a file that another process (CICFlowMeter) keeps
appending to. Reads each row exactly once by remembering the byte offset.

Handles: partial last line (waits for the newline), CRLF, UTF-8 BOM, a header
repeated mid-file, malformed rows (skipped + counted), truncation/rotation
(restarts from the top).
"""
import csv
import io
import os
from typing import List, Optional

import pandas as pd


class IncrementalCsvReader:
    def __init__(self, path: str, from_start: bool = True):
        self.path = path
        self.header: Optional[List[str]] = None
        self.offset = 0
        self.skipped = 0
        self._start_at_end = not from_start

    def _reset(self):
        self.header = None
        self.offset = 0

    def read_new(self, max_rows: int = 500) -> pd.DataFrame:
        """Returns up to max_rows new rows as a string DataFrame (may be empty)."""
        if not os.path.exists(self.path):
            return pd.DataFrame()
        size = os.path.getsize(self.path)
        if size < self.offset:           # truncated / rotated
            self._reset()

        with open(self.path, "rb") as f:
            if self._start_at_end:
                # Skip existing rows but still learn the header.
                first = f.readline()
                if not first.endswith(b"\n"):
                    return pd.DataFrame()
                self.header = self._parse(first)
                self.offset = size
                self._start_at_end = False
                return pd.DataFrame()
            f.seek(self.offset)
            chunk = f.read()

        last_nl = chunk.rfind(b"\n")
        if last_nl < 0:
            return pd.DataFrame()        # no complete line yet

        lines = chunk[: last_nl + 1].split(b"\n")[:-1]
        rows, consumed = [], 0
        for raw in lines:
            line_len = len(raw) + 1
            text = raw.decode("utf-8-sig", errors="replace").rstrip("\r")
            if not text.strip():
                consumed += line_len
                continue
            parsed = self._parse_text(text)
            if self.header is None:
                self.header = [c.strip() for c in parsed]
                consumed += line_len
                continue
            if [c.strip() for c in parsed] == self.header:   # repeated header
                consumed += line_len
                continue
            if len(parsed) != len(self.header):
                self.skipped += 1
                consumed += line_len
                continue
            if len(rows) >= max_rows:
                break
            rows.append(parsed)
            consumed += line_len

        self.offset += consumed
        if not rows:
            return pd.DataFrame(columns=self.header or [])
        return pd.DataFrame(rows, columns=self.header)

    @staticmethod
    def _parse_text(text: str) -> List[str]:
        return next(csv.reader(io.StringIO(text)), [])

    def _parse(self, raw: bytes) -> List[str]:
        return [c.strip() for c in self._parse_text(raw.decode("utf-8-sig", errors="replace").rstrip("\r\n"))]
