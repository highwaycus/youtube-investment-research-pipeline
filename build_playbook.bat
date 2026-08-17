from __future__ import annotations

import logging
import os
import json
import random
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from research import (
    RETRYABLE_TRANSCRIPT_STATUSES,
    build_channel_playbook,
    channel_playbook_is_stale,
    connect_db,
    extract_video_research,
    save_video_research,
    update_performance,
)

ROOT = Path(__file__).resolve().parent
YTDLP_EXE = ROOT / "yt-dlp.exe"


def run_ytdlp_json(args):
    if not YTDLP_EXE.exists():
        raise RuntimeError(
            "yt-dlp.exe is missing. Run update_ytdlp.bat once, then retry."
        )
    completed = subprocess.run(
        [str(YTDLP_EXE), "--no-warnings", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"yt-dlp.exe failed: {message[-3000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"yt-dlp.exe returned invalid JSON: {completed.stdout[-1000:]}"
        ) from exc


def is_restricted_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "members-only" in text or "available to this channel's members" in text


def is_completed_stream(detail) -> bool:
    return bool(detail.get("was_live")) or detail.get("live_status") in {
        "was_live", "post_live"
    }


def list_channel_streams(channel_id: str, since_date: str):
    url = f"https://www.youtube.com/channel/{channel_id}/streams"
    info = run_ytdlp_json([
        "--flat-playlist", "--playlist-end", "600", "--dump-single-json", url
    ])
    channel_name = info.get("channel") or info.get("uploader") or channel_id
    result = []
    entries = info.get("entries") or []
    if not entries:
        raise RuntimeError(
            f"yt-dlp.exe returned zero streams for {channel_id}. "
            "Run update_ytdlp.bat to refresh it."
        )
    old_streak = 0
    for index, entry in enumerate(entries, 1):
        video_id = entry.get("id")
        if not video_id:
            continue
        if entry.get("availability") in {
            "subscriber_only", "premium_only", "needs_auth", "private"
        }:
            logging.info("Skipping restricted stream %s", video_id)
            continue

        # YouTube's flat channel listing often omits upload dates. Fetch that
        # video's metadata only when needed, otherwise every entry would be
        # discarded and the backfill would incorrectly report zero videos.
        detail = entry
        if not is_completed_stream(entry) or not (
            entry.get("timestamp")
            or entry.get("release_timestamp")
            or entry.get("upload_date")
        ):
            try:
                detail = run_ytdlp_json([
                    "--skip-download", "--dump-single-json",
                    f"https://www.youtube.com/watch?v={video_id}",
                ]) or entry
            except Exception as exc:
                if is_restricted_error(exc):
                    logging.info("Skipping members-only stream %s", video_id)
                else:
                    logging.warning("Could not read metadata for %s: %s", video_id, exc)
                continue
            if index == 1 or index % 10 == 0:
                logging.info("Checked metadata for %d channel streams", index)

        if not is_completed_stream(detail):
            # Do not analyze upcoming or currently-running streams; they can
            # be picked up after the archive and transcript become available.
            continue

        stamp = detail.get("release_timestamp") or detail.get("timestamp")
        upload_date = detail.get("upload_date")
        if stamp:
            published = datetime.fromtimestamp(stamp, timezone.utc).date().isoformat()
        elif upload_date and len(upload_date) == 8:
            published = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
        else:
            logging.warning("No publication date for %s; skipping", video_id)
            continue

        if published < since_date:
            old_streak += 1
            # Channel tabs are reverse chronological. A small streak allows
            # for an occasional pinned/out-of-order video before stopping.
            if old_streak >= 5:
                break
            continue
        old_streak = 0
        result.append({
            "channel_id": channel_id,
            "channel_name": detail.get("channel") or detail.get("uploader") or channel_name,
            "video_id": video_id,
            "title": detail.get("title") or entry.get("title") or "Untitled",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "published_date": published,
            "description": detail.get("description") or entry.get("description") or "",
            "content_type": "stream",
        })
    return result


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(root, ".env"))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "backfill.log"), encoding="utf-8"),
        ],
    )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    channels = [x.strip() for x in os.getenv("YOUTUBE_CHANNEL_IDS", "").split(",") if x.strip()]
    if not channels:
        raise RuntimeError("YOUTUBE_CHANNEL_IDS is missing from .env")
    days = int(os.getenv("BACKFILL_DAYS", "183"))
    requested_max = int(os.getenv("MAX_BACKFILL_VIDEOS_PER_RUN", "5"))
    max_per_run = min(max(1, requested_max), 10)
    if requested_max > 10:
        logging.warning(
            "MAX_BACKFILL_VIDEOS_PER_RUN=%d is unsafe; limiting this run to 10",
            requested_max,
        )
    delay_min = max(0, int(os.getenv("BACKFILL_DELAY_MIN_SECONDS", "20")))
    delay_max = max(delay_min, int(os.getenv("BACKFILL_DELAY_MAX_SECONDS", "40")))
    since = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    conn = connect_db()
    candidates = []
    for channel_id in channels:
        logging.info("Listing six-month stream history for %s", channel_id)
        candidates.extend(list_channel_streams(channel_id, since))
    records = {
        row["video_id"]: {
            "status": row["transcript_status"],
            "attempts": int(row["transcript_attempts"]),
        }
        for row in conn.execute(
            "SELECT video_id, transcript_status, transcript_attempts FROM videos "
            "WHERE content_type = 'stream'"
        )
    }
    retryable = set(RETRYABLE_TRANSCRIPT_STATUSES)
    pending = [
        item for item in candidates
        if item["video_id"] not in records
        or records[item["video_id"]]["status"] in retryable
    ]
    # Rotate through the backlog instead of letting one repeatedly failing
    # video permanently block all later videos.
    pending.sort(key=lambda item: (
        records.get(item["video_id"], {}).get("attempts", 0),
        item["published_date"],
    ))
    selected = pending[:max_per_run]
    logging.info(
        "Found %d new/retryable streams; processing at most %d this run",
        len(pending), len(selected),
    )
    analyzed = 0
    stopped_for_block = False
    for index, item in enumerate(selected, 1):
        logging.info("[%d/%d] %s", index, len(selected), item["title"])
        data = extract_video_research(**item)
        save_video_research(conn, data)
        analyzed += 1
        if data["transcript_status"] == "blocked":
            stopped_for_block = True
            logging.error(
                "YouTube blocked transcript requests. Stopping immediately; "
                "remaining videos were not marked as processed. Wait several "
                "hours before running run_backfill.bat again."
            )
            break
        if index < len(selected) and delay_max:
            delay = random.uniform(delay_min, delay_max)
            logging.info("Waiting %.0f seconds before the next transcript request", delay)
            time.sleep(delay)
    evaluated = update_performance(conn)
    logging.info(
        "Backfill batch complete. Attempted %d; evaluated %d matured signals",
        analyzed, evaluated,
    )
    refreshed_statuses = {
        row["video_id"]: row["transcript_status"]
        for row in conn.execute(
            "SELECT video_id, transcript_status FROM videos "
            "WHERE content_type = 'stream'"
        )
    }
    remaining = sum(
        1 for item in candidates
        if item["video_id"] not in refreshed_statuses
        or refreshed_statuses[item["video_id"]] in retryable
    )
    if stopped_for_block:
        logging.info(
            "%d streams remain. Do not retry immediately; wait several hours.",
            remaining,
        )
    elif remaining:
        logging.info(
            "%d streams remain. Wait at least 15-30 minutes before running "
            "run_backfill.bat again.",
            remaining,
        )
    elif analyzed or channel_playbook_is_stale():
        logging.info("Historical stream backfill is complete; updating channel playbook.")
        build_channel_playbook(conn, days=days)
    else:
        logging.info("Historical stream backfill and channel playbook are already current.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Backfill failed")
        raise SystemExit(1)
