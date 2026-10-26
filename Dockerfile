FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[openai,ui]"

COPY data/policies ./data/policies
COPY app ./app

RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app/data
USER appuser

ENV OPSPILOT_POLICIES_DIR=/app/data/policies \
    OPSPILOT_TICKET_DB_PATH=/app/data/tickets.sqlite3

EXPOSE 8000
CMD ["uvicorn", "opspilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
