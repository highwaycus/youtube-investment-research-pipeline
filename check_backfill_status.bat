from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from research import _research_cutoff, build_channel_playbook, connect_db
from signal_model import train_signal_model


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(root, ".env"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    conn = connect_db()
    days = int(os.getenv("BACKFILL_DAYS", "183"))
    build_channel_playbook(conn, days=days)
    train_signal_model(conn, cutoff=_research_cutoff(conn, days))
    logging.info("Channel playbook is ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Playbook build failed")
        raise SystemExit(1)
