from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "auto_backfill_state.json"
LOG_PATH = ROOT / "logs" / "auto_backfill.log"
COOLDOWN_HOURS = 24

BLOCK_MARKERS = (
    "youtube blocked transcript requests",
    "youtube is blocking requests from your ip",
    "[blocked]",
)
COMPLETE_MARKERS = (
    "historical stream backfill is complete",
    "historical stream backfill and channel playbook are already current",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(**values) -> None:
    state = load_state()
    state.update(values)
    state["updated_at"] = utc_now().isoformat()
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(STATE_PATH)


def display_local(instant: datetime) -> str:
    return instant.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def run_backfill() -> tuple[int, str]:
    LOG_PATH.parent.mkdir(exist_ok=True)
    command = [sys.executable, str(ROOT / "backfill.py")]
    captured: list[str] = []
    with LOG_PATH.open("a", encoding="utf-8") as log:
        header = f"\n=== Automatic run {utc_now().isoformat()} ===\n"
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            captured.append(line)
        return process.wait(), "".join(captured)


def main() -> int:
    state = load_state()
    if state.get("completed"):
        print(
            "Historical backfill is already complete. "
            "No YouTube or OpenAI request was sent."
        )
        return 0

    now = utc_now()
    cooldown_until = parse_utc(state.get("cooldown_until"))
    if cooldown_until and now < cooldown_until:
        print(
            "Automatic backfill is cooling down after a YouTube block. "
            f"Next allowed attempt: {display_local(cooldown_until)}."
        )
        print("No YouTube or OpenAI request was sent.")
        return 0

    code, output = run_backfill()
    lowered = output.lower()

    if any(marker in lowered for marker in BLOCK_MARKERS):
        retry_at = utc_now() + timedelta(hours=COOLDOWN_HOURS)
        save_state(
            completed=False,
            cooldown_until=retry_at.isoformat(),
            last_result="blocked",
        )
        print(
            f"YouTube block detected. Automatic attempts are paused until "
            f"{display_local(retry_at)}."
        )
        return 0

    if any(marker in lowered for marker in COMPLETE_MARKERS):
        save_state(
            completed=True,
            cooldown_until=None,
            last_result="complete",
        )
        print(
            "Historical backfill is complete. Future scheduled launches will "
            "exit without sending requests."
        )
        return 0

    if code != 0:
        save_state(last_result=f"error_{code}")
        print(
            "Backfill ended with an error. The scheduler may try again at its "
            "next hourly run."
        )
        return code

    save_state(
        completed=False,
        cooldown_until=None,
        last_result="batch_complete",
    )
    print("This batch finished. The next batch may run in one hour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
