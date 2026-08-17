from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "research.db"


def main() -> int:
    if not DB_PATH.exists():
        print("research.db not found. Run run_backfill.bat first.")
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT transcript_status, COUNT(*) AS count
        FROM videos
        WHERE content_type = 'stream'
        GROUP BY transcript_status
        ORDER BY transcript_status"""
    ).fetchall()
    counts = {row["transcript_status"]: int(row["count"]) for row in rows}
    retryable = sum(
        counts.get(status, 0)
        for status in ("unavailable", "blocked", "temporary_error")
    )
    coverage = conn.execute(
        """SELECT MIN(published_date) AS first_date,
                  MAX(published_date) AS last_date,
                  COUNT(*) AS streams
        FROM videos
        WHERE content_type = 'stream' AND transcript_status = 'available'"""
    ).fetchone()
    signals = conn.execute(
        """SELECT COUNT(*)
        FROM signals s JOIN videos v ON v.video_id = s.video_id
        WHERE v.content_type = 'stream' AND v.transcript_status = 'available'"""
    ).fetchone()[0]

    print("Transcript status")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    print(f"  retryable remaining: {retryable}")
    print(
        "Successful coverage: "
        f"{coverage['streams']} streams, "
        f"{coverage['first_date'] or '-'} to {coverage['last_date'] or '-'}"
    )
    print(f"Extracted signals: {signals}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
