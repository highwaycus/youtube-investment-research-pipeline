# Comma-separated YouTube channel IDs (the UC... value, not @handle)
YOUTUBE_CHANNEL_IDS=UCxxxxxxxxxxxxxxxxxxxxxx,UCyyyyyyyyyyyyyyyyyyyyyy

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5-mini

# Email (Gmail example; use a Google App Password, not your normal password)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_address@gmail.com
SMTP_PASSWORD=your_16_character_app_password
EMAIL_FROM=your_address@gmail.com
EMAIL_TO=your_address@gmail.com

# Optional behavior
MAX_NEW_VIDEOS_PER_CHANNEL=1
LATEST_SIGNAL_DAYS=7

# Historical research (run_backfill.bat); resumable to control cost
BACKFILL_DAYS=183
MAX_BACKFILL_VIDEOS_PER_RUN=5
BACKFILL_DELAY_MIN_SECONDS=20
BACKFILL_DELAY_MAX_SECONDS=40
