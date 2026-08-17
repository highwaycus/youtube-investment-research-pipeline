# Credentials and local configuration
.env
.env.*
!.env.example

# Python
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

# Private portfolio data
portfolio.json

# Generated research data and model artifacts
*.db
channel_playbook.json
fundamentals_cache.json
signal_model.joblib
signal_model_report.json
state.json
delivery_state.json
auto_backfill_state.json

# Runtime output
logs/
reports/

# Local packages and editor files
*.zip
.idea/
.vscode/
.DS_Store
Thumbs.db
